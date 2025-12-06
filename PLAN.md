# Parallel Task Execution, Timeout Handling, and Task Queue Design

This document proposes a unified design for three interconnected features that fundamentally improve MASEval's task execution architecture:

1. **Parallel Processing** - Concurrent task execution via asyncio or threading
2. **Timeout Handling** - Per-task timeout with graceful failure recording
3. **Task Queue** - Callback-driven task scheduling for adaptive testing

All three features directly impact the `Benchmark.run()` task loop and should be designed together.

---

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Key Architectural Changes](#key-architectural-changes)
3. [Feature 1: Parallel Processing](#feature-1-parallel-processing)
4. [Feature 2: Timeout Handling & TaskProtocol](#feature-2-timeout-handling--taskprotocol)
5. [Feature 3: TaskQueue](#feature-3-taskqueue)
6. [Unified Design Proposal](#unified-design-proposal)
7. [Implementation Phases](#implementation-phases)
8. [Risks and Mitigations](#risks-and-mitigations)

---

## Current Architecture Analysis

### The Run Loop (`benchmark.py` lines 990-1330)

The current execution model is strictly sequential:

```python
def run(self, tasks: ...):
    for task_idx, (task, agent_data) in enumerate(zip(self.tasks, agent_data_list)):
        for repeat_idx in range(self.n_task_repeats):
            # Setup
            environment = self.setup_environment(agent_data, task)
            # ... more setup

            # Execute
            final_answers = self.execution_loop(agents_to_run, task, environment, user)

            # Evaluate
            eval_results = self.evaluate(...)

            # Store
            self.reports.append(report)
```

### Key Observations

1. **Sequential by Design**: No parallelism, no timeouts, no queue abstraction
2. **Callback System**: Already has lifecycle hooks (`on_task_start`, `on_task_repeat_end`, etc.) but callbacks cannot influence task ordering
3. **Component Registry**: Per-task-repetition component tracking with `register()` / `clear_registry()`
4. **Error Handling**: Comprehensive status enum (`TaskExecutionStatus`) with graceful failure paths
5. **Agent Adapters**: Framework-specific adapters (smolagents, langgraph) that may or may not be async-native
6. **Model Adapters**: API clients that are inherently I/O-bound

### Critical Dependencies for Concurrency

| Component                   | Thread-Safety  | Async-Native | Notes                            |
| --------------------------- | -------------- | ------------ | -------------------------------- |
| `Benchmark.reports`         | ❌ List append | N/A          | Needs synchronization            |
| `Benchmark._trace_registry` | ❌ Dict        | N/A          | Per-task, but needs isolation    |
| `CallbackHandler`           | ❌             | N/A          | Callbacks may not be thread-safe |
| `SmolAgentAdapter`          | ✅ (stateless) | ❌           | Uses sync `agent.run()`          |
| `LangGraphAgentAdapter`     | ✅ (stateless) | ⚠️ Partial   | LangGraph has `ainvoke()`        |
| `GoogleGenAIModelAdapter`   | ✅             | ⚠️ Partial   | Google client has async methods  |

---

## Key Architectural Changes

This section summarizes the major architectural decisions made during planning.

### 1. Extract `ComponentRegistry` from `Benchmark`

**Problem**: The component registry logic (~150 lines) is mixed with benchmark orchestration. Adding thread-local handling will make it worse.

**Solution**: Extract into a dedicated `ComponentRegistry` class in `maseval/core/registry.py`.

```python
# maseval/core/registry.py

import threading
from typing import Dict, Any, Optional
from datetime import datetime

from .tracing import TraceableMixin
from .config import ConfigurableMixin


class ComponentRegistry:
    """Thread-safe registry for tracking components during task execution.

    Each thread gets its own isolated registry state, enabling parallel
    task execution without cross-contamination. The registry tracks both
    Traceable and Configurable components for comprehensive data collection.

    Usage:
        registry = ComponentRegistry()

        # Register components (thread-local)
        registry.register("agents", "orchestrator", agent_adapter)
        registry.register("environment", "env", environment)

        # Collect data
        traces = registry.collect_traces()
        configs = registry.collect_configs()

        # Clear for next task
        registry.clear()
    """

    def __init__(self, benchmark_config: Optional[Dict[str, Any]] = None):
        """Initialize the registry.

        Args:
            benchmark_config: Benchmark-level configuration to include in
                collect_configs() output. This is shared (not thread-local).
        """
        self._local = threading.local()
        self._benchmark_config = benchmark_config or {}

    # --- Thread-local state properties ---

    @property
    def _trace_registry(self) -> Dict[str, TraceableMixin]:
        if not hasattr(self._local, 'trace_registry'):
            self._local.trace_registry = {}
        return self._local.trace_registry

    @property
    def _component_id_map(self) -> Dict[int, str]:
        if not hasattr(self._local, 'component_id_map'):
            self._local.component_id_map = {}
        return self._local.component_id_map

    @property
    def _config_registry(self) -> Dict[str, ConfigurableMixin]:
        if not hasattr(self._local, 'config_registry'):
            self._local.config_registry = {}
        return self._local.config_registry

    @property
    def _config_component_id_map(self) -> Dict[int, str]:
        if not hasattr(self._local, 'config_component_id_map'):
            self._local.config_component_id_map = {}
        return self._local.config_component_id_map

    # --- Public API ---

    def register(self, category: str, name: str, component: TraceableMixin) -> TraceableMixin:
        """Register a component for trace and config collection.

        Args:
            category: Component category (e.g., "agents", "models", "environment")
            name: Unique identifier within the category
            component: Component instance (must be TraceableMixin)

        Returns:
            The component (for chaining)

        Raises:
            ValueError: If component already registered under a different key
        """
        component_id = id(component)
        key = f"{category}:{name}"

        # Check for duplicate registration under different key
        if component_id in self._component_id_map:
            existing_key = self._component_id_map[component_id]
            if existing_key != key:
                raise ValueError(
                    f"Component already registered as '{existing_key}', "
                    f"cannot re-register as '{key}'."
                )
            return component  # Idempotent

        # Register for tracing
        self._trace_registry[key] = component
        self._component_id_map[component_id] = key

        # Also register for config if supported
        if isinstance(component, ConfigurableMixin):
            self._config_registry[key] = component
            self._config_component_id_map[component_id] = key

        return component

    def clear(self) -> None:
        """Clear all registrations for the current thread."""
        self._trace_registry.clear()
        self._component_id_map.clear()
        self._config_registry.clear()
        self._config_component_id_map.clear()

    def collect_traces(self) -> Dict[str, Any]:
        """Collect execution traces from all registered components."""
        traces: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "thread_id": threading.current_thread().ident,
                "total_components": len(self._trace_registry),
            },
            "agents": {},
            "models": {},
            "tools": {},
            "simulators": {},
            "callbacks": {},
            "environment": None,
            "user": None,
            "other": {},
        }

        for key, component in self._trace_registry.items():
            category, comp_name = key.split(":", 1)
            try:
                component_traces = component.gather_traces()
                if "name" not in component_traces:
                    component_traces["name"] = comp_name

                if category == "environment":
                    traces["environment"] = component_traces
                elif category == "user":
                    traces["user"] = component_traces
                else:
                    if category not in traces:
                        traces[category] = {}
                    traces[category][comp_name] = component_traces
            except Exception as e:
                error_info = {"error": str(e), "error_type": type(e).__name__}
                if category in ("environment", "user"):
                    traces[category] = error_info
                else:
                    if category not in traces:
                        traces[category] = {}
                    traces[category][comp_name] = error_info

        return traces

    def collect_configs(self) -> Dict[str, Any]:
        """Collect configuration from all registered components."""
        configs: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "thread_id": threading.current_thread().ident,
                "total_components": len(self._config_registry),
            },
            "agents": {},
            "models": {},
            "tools": {},
            "simulators": {},
            "callbacks": {},
            "environment": None,
            "user": None,
            "other": {},
            "benchmark": self._benchmark_config,
        }

        for key, component in self._config_registry.items():
            category, comp_name = key.split(":", 1)
            try:
                component_config = component.gather_config()
                if "name" not in component_config:
                    component_config["name"] = comp_name

                if category == "environment":
                    configs["environment"] = component_config
                elif category == "user":
                    configs["user"] = component_config
                else:
                    if category not in configs:
                        configs[category] = {}
                    configs[category][comp_name] = component_config
            except Exception as e:
                error_info = {"error": str(e), "error_type": type(e).__name__}
                if category in ("environment", "user"):
                    configs[category] = error_info
                else:
                    if category not in configs:
                        configs[category] = {}
                    configs[category][comp_name] = error_info

        return configs
```

**Benchmark integration** (delegation pattern):

```python
class Benchmark:
    def __init__(self, ...):
        # ...
        self._registry = ComponentRegistry(
            benchmark_config=gather_benchmark_config()
        )

    def register(self, category: str, name: str, component: TraceableMixin) -> TraceableMixin:
        """Register a component. Delegates to internal registry."""
        return self._registry.register(category, name, component)

    def clear_registry(self) -> None:
        """Clear registry after task repetition."""
        self._registry.clear()

    def collect_all_traces(self) -> Dict[str, Any]:
        """Collect traces. Delegates to internal registry."""
        return self._registry.collect_traces()

    def collect_all_configs(self) -> Dict[str, Any]:
        """Collect configs. Delegates to internal registry."""
        return self._registry.collect_configs()
```

**Benefits**:

- Single Responsibility: Benchmark orchestrates, Registry tracks components
- Testability: Registry can be unit tested in isolation
- Clarity: Thread-local complexity encapsulated in one place
- Zero API changes: Users still call `benchmark.register(...)`

### 2. Threading over asyncio

**Decision**: Use `ThreadPoolExecutor` for parallel task execution.

**Rationale**:

- No user code changes required (async would require rewriting `run_agents()`)
- Works with all agent frameworks (smolagents is sync-only)
- Same I/O concurrency benefits for LLM API calls
- Future-proof: Python's GIL removal will make threading even more powerful

### 3. MASEval-Managed Callback Thread Safety

**Decision**: MASEval serializes all callback invocations with an internal lock.

**Rationale**:

- Users don't need to think about thread safety
- Negligible performance cost (callbacks are fast)
- Prevents subtle race condition bugs

```python
class Benchmark:
    def __init__(self, ...):
        self._callback_lock = threading.Lock()

    def _invoke_callbacks(self, method_name: str, *args, **kwargs):
        with self._callback_lock:
            for cb in self.callbacks:
                getattr(cb, method_name)(*args, **kwargs)
```

### 4. Cooperative Timeout with Hard Backstop

**Decision**: Use cooperative checkpoint-based timeout with a hard timeout fallback.

**Rationale**:

- Cross-platform (signal-based only works on Unix)
- Works in threads (signals only work in main thread)
- Clean interruption at defined checkpoints
- Hard timeout as last resort for misbehaving code

**Limitation**: Python threads cannot be forcibly killed. Timeout is "best effort."

---

## Feature 1: Parallel Processing

### Decision: Threading with `ThreadPoolExecutor`

We use **threading** (not asyncio) for parallel task execution.

#### Why Threading Over asyncio

| Consideration           | Threading                     | asyncio                              |
| ----------------------- | ----------------------------- | ------------------------------------ |
| User code changes       | None                          | Must rewrite `run_agents()` as async |
| Agent framework support | All (smolagents is sync-only) | Only async-native frameworks         |
| API signature           | Unchanged                     | Breaking (`async def run()`)         |
| Mental model            | Familiar to most developers   | Requires async expertise             |
| Future GIL removal      | Benefits automatically        | No additional benefit                |

**asyncio would require**:

- All user-implemented methods become `async def`
- Wrapper code for sync agent frameworks (smolagents)
- Breaking API changes throughout

**Threading provides**:

- Zero user code changes
- Works with all agent frameworks today
- Same I/O concurrency benefits (LLM API calls)
- Future-proof: when Python removes the GIL, threading will gain true parallelism

#### Implementation: `ThreadPoolExecutor`

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def run(self, tasks, max_workers: int = 1):  # max_workers=1 = sequential (default)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for task, agent_data in zip(self.tasks, agent_data_list):
            future = executor.submit(self._run_single_task, task, agent_data)
            futures[future] = task

        for future in as_completed(futures):
            report = future.result()
            self._append_report_safe(report)  # Thread-safe
```

**Key design points**:

1. **Backwards Compatible**: `max_workers=1` maintains current sequential behavior
2. **Framework Agnostic**: Works with sync agent frameworks (smolagents)
3. **I/O Parallelism**: Multiple LLM API calls can happen concurrently
4. **Opt-in**: Users explicitly enable parallelism

#### Thread-Local Component Registry

The component registry is already cleared after each task repetition. For parallel execution, we make it thread-local so concurrent tasks don't share registries:

```python
import threading

class Benchmark:
    def __init__(self, ...):
        self._local = threading.local()

    @property
    def _trace_registry(self):
        if not hasattr(self._local, 'trace_registry'):
            self._local.trace_registry = {}
        return self._local.trace_registry

    @property
    def _component_id_map(self):
        if not hasattr(self._local, 'component_id_map'):
            self._local.component_id_map = {}
        return self._local.component_id_map

    # Same pattern for _config_registry, _config_component_id_map
```

This is the correct design because:

- Each task repetition runs in one thread
- Registries are ephemeral (cleared after each repetition via `clear_registry()`)
- No cross-task state sharing is intended

#### Thread-Safe Report Collection

```python
import threading

class Benchmark:
    def __init__(self, ...):
        self._reports_lock = threading.Lock()

    def _append_report_safe(self, report):
        with self._reports_lock:
            self.reports.append(report)
```

#### Thread-Safe Callback Invocation

MASEval serializes all callback invocations internally, so **users don't need to implement thread-safe callbacks**:

```python
class Benchmark:
    def __init__(self, ...):
        self._callback_lock = threading.Lock()

    def _invoke_callbacks(self, method_name: str, *args, **kwargs):
        """Invoke a callback method on all registered callbacks (thread-safe)."""
        with self._callback_lock:
            for cb in self.callbacks:
                getattr(cb, method_name)(*args, **kwargs)
```

**User impact**: None. Users write callbacks exactly as they do today:

```python
class MyCallback(BenchmarkCallback):
    def __init__(self):
        self.count = 0  # No lock needed!

    def on_task_repeat_end(self, benchmark, report):
        self.count += 1  # Safe because MASEval serializes calls
```

This approach:

- Eliminates thread-safety burden on users
- Has negligible performance cost (callbacks are fast)
- Prevents an entire class of subtle bugs

````

---

## Feature 2: Timeout Handling & TaskProtocol

### Design Goal

Enable per-task timeout configuration, capturing partial results on timeout.

### The `TaskProtocol` Concept

A `TaskProtocol` dataclass defines task-level execution parameters. It's attached to `Task` but describes how MASEval should run the task, not task content.

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class TimeoutAction(Enum):
    """What to do when a timeout occurs."""
    SKIP = "skip"           # Mark as timed out, continue to next task
    RETRY = "retry"         # Retry once with same timeout
    EXTEND = "extend"       # Double timeout and retry


@dataclass
class TaskProtocol:
    """Configuration for how MASEval executes a task.

    This is a data container for execution parameters, separate from
    task content (query, environment_data, etc.). It controls the
    interface between the task and MASEval's execution engine.

    Attributes:
        timeout_seconds: Maximum execution time for this task. None means no timeout.
        timeout_action: Action to take when timeout occurs.
        max_retries: Maximum retry attempts for transient failures (not timeouts).
        priority: Execution priority (higher = sooner). Used by adaptive task queues.
        tags: Arbitrary tags for filtering or grouping tasks.
    """
    timeout_seconds: Optional[float] = None
    timeout_action: TimeoutAction = TimeoutAction.SKIP
    max_retries: int = 0
    priority: int = 0
    tags: dict = field(default_factory=dict)
````

### Attaching Protocol to Task

```python
@dataclass
class Task:
    query: str
    id: UUID = field(default_factory=uuid4)
    environment_data: Dict[str, Any] = field(default_factory=dict)
    evaluation_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # New: execution protocol
    protocol: TaskProtocol = field(default_factory=TaskProtocol)
```

### Timeout Implementation Strategies

#### Strategy A: `concurrent.futures` with Timeout

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def _run_task_with_timeout(self, task, agent_data, timeout: Optional[float]):
    """Run a single task with optional timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(self._run_single_task_inner, task, agent_data)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            # Attempt to cancel (may not stop running code)
            future.cancel()
            return self._create_timeout_report(task)
```

**Problem**: `future.cancel()` doesn't actually stop running Python code. The task continues executing in the background.

#### Strategy B: `multiprocessing` with Termination

```python
from multiprocessing import Process, Queue

def _run_task_with_timeout(self, task, agent_data, timeout):
    result_queue = Queue()
    process = Process(target=self._run_in_process, args=(task, agent_data, result_queue))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()  # Actually kills the task
        return self._create_timeout_report(task)

    return result_queue.get()
```

**Problem**: Process isolation means no shared state. Components can't be registered, traces can't be collected incrementally.

#### Strategy C: Signal-Based Timeout (Unix only)

```python
import signal

class TimeoutException(Exception):
    pass

def _run_task_with_timeout(self, task, agent_data, timeout):
    def handler(signum, frame):
        raise TimeoutException()

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(timeout))
    try:
        return self._run_single_task_inner(task, agent_data)
    except TimeoutException:
        return self._create_timeout_report(task)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
```

**Problem**: Only works on Unix. Doesn't work in threads (signal only works in main thread).

#### Strategy D: Cooperative Timeout with Checkpoints (Recommended)

The cleanest approach that works cross-platform and with threads is **cooperative timeout checking**:

```python
import time
import threading

class TaskContext:
    """Execution context passed to user code for timeout checking."""

    def __init__(self, deadline: Optional[float] = None):
        self._deadline = deadline
        self._start_time = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def remaining(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return max(0, self._deadline - self.elapsed)

    @property
    def is_expired(self) -> bool:
        return self._deadline is not None and self.elapsed >= self._deadline

    def check_timeout(self):
        """Raise TimeoutError if deadline exceeded. Call at checkpoints."""
        if self.is_expired:
            raise TaskTimeoutError(f"Task exceeded {self._deadline}s deadline")
```

**Usage in `run_agents()`**:

```python
def run_agents(self, agents, task, environment, query, context: TaskContext) -> Any:
    for step in range(self.max_steps):
        context.check_timeout()  # Cooperative checkpoint
        result = agents[0].run(query)
        # ...
```

**Hybrid with Hard Timeout**: Combine cooperative checking with a hard timeout fallback:

```python
def _run_task_with_timeout(self, task, agent_data, timeout):
    context = TaskContext(deadline=timeout)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(self._run_single_task_inner, task, agent_data, context)
        try:
            # Hard timeout as backstop
            return future.result(timeout=timeout + 5)  # Grace period
        except TimeoutError:
            return self._create_timeout_report(task, partial_traces=context.collected_traces)
```

### New Exception Type

```python
class TaskTimeoutError(MASEvalError):
    """Task execution exceeded configured timeout.

    This is classified as TASK_TIMEOUT in benchmark results, separate from
    other error types. Timeout is neither agent's fault nor infrastructure's
    fault—it's a resource constraint.
    """

    def __init__(self, message: str, elapsed: float, timeout: float, partial_traces: Optional[Dict] = None):
        super().__init__(message, component="timeout")
        self.elapsed = elapsed
        self.timeout = timeout
        self.partial_traces = partial_traces or {}
```

### New Status

```python
class TaskExecutionStatus(Enum):
    SUCCESS = "success"
    AGENT_ERROR = "agent_error"
    ENVIRONMENT_ERROR = "environment_error"
    USER_ERROR = "user_error"
    UNKNOWN_EXECUTION_ERROR = "unknown_execution_error"
    EVALUATION_FAILED = "evaluation_failed"
    SETUP_FAILED = "setup_failed"
    TASK_TIMEOUT = "task_timeout"  # NEW
```

---

## Feature 3: TaskQueue

### Design Goal

Replace the static `for task in tasks` loop with a queue abstraction that enables:

1. Dynamic task ordering
2. Callback-driven scheduling (adaptive testing)
3. Priority-based execution
4. Conditional task skipping

### The `TaskQueue` Interface

```python
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple

class TaskQueue(ABC):
    """Abstract base for task scheduling strategies."""

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[Task, Dict]]:
        """Yield (task, agent_data) pairs in execution order."""
        pass

    def on_task_complete(self, task: Task, report: Dict) -> None:
        """Called after each task completes. Override for adaptive behavior."""
        pass

    def should_continue(self) -> bool:
        """Whether to continue processing. Default: True until queue exhausted."""
        return True


class SequentialQueue(TaskQueue):
    """Default: execute tasks in order (current behavior)."""

    def __init__(self, tasks: TaskCollection, agent_data_list: List[Dict]):
        self._tasks = list(zip(tasks, agent_data_list))
        self._index = 0

    def __iter__(self):
        for task, agent_data in self._tasks:
            yield task, agent_data


class PriorityQueue(TaskQueue):
    """Execute tasks by priority (from TaskProtocol.priority)."""

    def __init__(self, tasks: TaskCollection, agent_data_list: List[Dict]):
        paired = list(zip(tasks, agent_data_list))
        # Sort by priority descending
        self._tasks = sorted(paired, key=lambda x: x[0].protocol.priority, reverse=True)

    def __iter__(self):
        for task, agent_data in self._tasks:
            yield task, agent_data


class AdaptiveQueue(TaskQueue):
    """Adaptive testing: adjust task order based on results.

    Example: Item Response Theory (IRT) based testing that estimates
    agent difficulty and selects optimally informative tasks.
    """

    def __init__(self, tasks: TaskCollection, agent_data_list: List[Dict]):
        self._pending = list(zip(tasks, agent_data_list))
        self._completed = []
        self._agent_ability_estimate = 0.0

    def __iter__(self):
        while self._pending and self.should_continue():
            # Select next task based on estimated ability
            next_task = self._select_next_task()
            if next_task:
                yield next_task

    def _select_next_task(self) -> Optional[Tuple[Task, Dict]]:
        """Select task that maximizes information gain."""
        if not self._pending:
            return None

        # IRT-based selection (simplified)
        best_idx = 0
        best_info = 0

        for idx, (task, _) in enumerate(self._pending):
            difficulty = task.metadata.get("difficulty", 0.5)
            # Fisher information at current ability estimate
            info = self._fisher_information(difficulty, self._agent_ability_estimate)
            if info > best_info:
                best_info = info
                best_idx = idx

        return self._pending.pop(best_idx)

    def on_task_complete(self, task: Task, report: Dict) -> None:
        """Update ability estimate based on task result."""
        self._completed.append((task, report))
        self._update_ability_estimate()

    def _update_ability_estimate(self):
        """Bayesian update of ability estimate."""
        # Implementation depends on IRT model
        pass

    def should_continue(self) -> bool:
        """Stop when estimate is precise enough."""
        return len(self._completed) < 50  # Example stopping rule
```

### Integration with `Benchmark.run()`

```python
def run(
    self,
    tasks: Union[Task, TaskCollection, Iterable[Union[Task, dict]]],
    queue: Optional[TaskQueue] = None,
    max_workers: int = 1,
) -> List[Dict[str, Any]]:
    # Normalize tasks
    task_collection = self._normalize_tasks(tasks)
    agent_data_list = self._normalize_agent_data(task_collection)

    # Create queue (default: sequential)
    if queue is None:
        queue = SequentialQueue(task_collection, agent_data_list)

    # Callbacks
    for cb in self.callbacks:
        cb.on_run_start(self)

    # Execute via queue
    if max_workers == 1:
        self._run_sequential(queue)
    else:
        self._run_parallel(queue, max_workers)

    # Callbacks
    for cb in self.callbacks:
        cb.on_run_end(self, self.reports)

    return self.reports

def _run_sequential(self, queue: TaskQueue):
    for task, agent_data in queue:
        for repeat_idx in range(self.n_task_repeats):
            report = self._execute_single_repetition(task, agent_data, repeat_idx)
            self.reports.append(report)
            queue.on_task_complete(task, report)

            if not queue.should_continue():
                return
```

### Callback Integration for Adaptive Testing

The existing `BenchmarkCallback` can be extended:

```python
class BenchmarkCallback(ABC, TraceableMixin):
    # ... existing methods ...

    def on_task_selected(self, benchmark: "Benchmark", task: "Task", queue: "TaskQueue"):
        """Called when TaskQueue selects the next task to run."""
        pass

    def on_queue_decision(self, benchmark: "Benchmark", queue: "TaskQueue", should_continue: bool):
        """Called when TaskQueue makes a continue/stop decision."""
        pass
```

---

## Unified Design Proposal

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Benchmark.run()                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐                                          │
│  │    TaskQueue      │ ← Adaptive/Priority/Sequential           │
│  │  (iterator)       │                                          │
│  └────────┬──────────┘                                          │
│           │ yields (Task, agent_data)                           │
│           ▼                                                     │
│  ┌────────────────────────────────────────────────────────────┐│
│  │              ThreadPoolExecutor (max_workers)               ││
│  │  ┌──────────────────────────────────────────────────────┐  ││
│  │  │  Worker Thread 1                                     │  ││
│  │  │  ┌─────────────────────────────────────────────────┐ │  ││
│  │  │  │ TaskContext (deadline, checkpoints)             │ │  ││
│  │  │  │ ┌─────────────────────────────────────────────┐ │ │  ││
│  │  │  │ │ setup → execution_loop → evaluate           │ │ │  ││
│  │  │  │ │ (Task.protocol.timeout_seconds)             │ │ │  ││
│  │  │  │ └─────────────────────────────────────────────┘ │ │  ││
│  │  │  └─────────────────────────────────────────────────┘ │  ││
│  │  └──────────────────────────────────────────────────────┘  ││
│  │  ┌──────────────────────────────────────────────────────┐  ││
│  │  │  Worker Thread 2 ...                                 │  ││
│  │  └──────────────────────────────────────────────────────┘  ││
│  └────────────────────────────────────────────────────────────┘│
│           │                                                     │
│           ▼ reports                                            │
│  ┌───────────────────┐                                          │
│  │  Thread-Safe      │                                          │
│  │  Report Collector │                                          │
│  └───────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Complete `run()` Implementation

```python
def run(
    self,
    tasks: Union[Task, TaskCollection, Iterable[Union[Task, dict]]],
    queue: Optional[TaskQueue] = None,
    max_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Run benchmark with parallel processing, timeouts, and adaptive scheduling.

    Args:
        tasks: Tasks to execute.
        queue: Task scheduling strategy. Default: SequentialQueue.
        max_workers: Maximum parallel task executions. Default: 1 (sequential).

    Returns:
        List of report dictionaries.
    """
    # Normalize inputs
    self.tasks = self._normalize_tasks(tasks)
    agent_data_list = self._normalize_agent_data()

    # Create queue
    if queue is None:
        queue = SequentialQueue(self.tasks, agent_data_list)

    # Clear reports
    self.reports = []
    self._reports_lock = threading.Lock()

    # Run start callbacks
    for cb in self.callbacks:
        cb.on_run_start(self)

    # Execute
    if max_workers == 1:
        self._run_sequential(queue)
    else:
        self._run_parallel(queue, max_workers)

    # Run end callbacks
    for cb in self.callbacks:
        cb.on_run_end(self, self.reports)

    return self.reports

def _run_sequential(self, queue: TaskQueue):
    """Sequential execution with timeout support."""
    for task, agent_data in queue:
        for cb in self.callbacks:
            cb.on_task_start(self, task)

        for repeat_idx in range(self.n_task_repeats):
            report = self._execute_task_repetition(task, agent_data, repeat_idx)
            self._append_report_safe(report)
            queue.on_task_complete(task, report)

        for cb in self.callbacks:
            cb.on_task_end(self, task, self._last_report_for_task(task))

        if not queue.should_continue():
            break

def _run_parallel(self, queue: TaskQueue, max_workers: int):
    """Parallel execution with timeout support."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Submit initial batch
        for task, agent_data in queue:
            for repeat_idx in range(self.n_task_repeats):
                future = executor.submit(
                    self._execute_task_repetition,
                    task, agent_data, repeat_idx
                )
                futures[future] = (task, repeat_idx)

            if len(futures) >= max_workers * 2:
                break  # Don't over-submit

        # Process completions and submit more
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            for future in done:
                task, repeat_idx = futures.pop(future)
                try:
                    report = future.result()
                except Exception as e:
                    report = self._create_error_report(task, repeat_idx, e)

                self._append_report_safe(report)
                queue.on_task_complete(task, report)

                # Callbacks serialized internally (thread-safe for users)
                self._invoke_callbacks('on_task_repeat_end', self, report)

                if not queue.should_continue():
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    return

            # Submit more work
            try:
                task, agent_data = next(iter(queue))
                for repeat_idx in range(self.n_task_repeats):
                    future = executor.submit(
                        self._execute_task_repetition,
                        task, agent_data, repeat_idx
                    )
                    futures[future] = (task, repeat_idx)
            except StopIteration:
                pass

def _execute_task_repetition(
    self,
    task: Task,
    agent_data: Dict[str, Any],
    repeat_idx: int,
) -> Dict[str, Any]:
    """Execute a single task repetition with timeout handling."""
    timeout = task.protocol.timeout_seconds
    context = TaskContext(deadline=timeout)

    # Thread-local registry for this execution
    local_registry = {}

    try:
        # Setup
        environment = self.setup_environment(agent_data, task)
        user = self.setup_user(agent_data, environment, task)
        agents_to_run, agents_dict = self.setup_agents(agent_data, environment, task, user)
        evaluators = self.setup_evaluators(environment, task, agents_to_run, user)

        # Register components (thread-local)
        local_registry.update(self._register_components(environment, user, agents_dict))

        # Execute with timeout checking
        context.check_timeout()
        final_answer = self.execution_loop(agents_to_run, task, environment, user, context)

        # Collect traces
        traces = self._collect_traces(local_registry)
        configs = self._collect_configs(local_registry)

        # Evaluate
        context.check_timeout()
        eval_results = self.evaluate(evaluators, agents_dict, final_answer, traces)

        return {
            "task_id": str(task.id),
            "repeat_idx": repeat_idx,
            "status": TaskExecutionStatus.SUCCESS.value,
            "traces": traces,
            "config": configs,
            "eval": eval_results,
        }

    except TaskTimeoutError as e:
        return {
            "task_id": str(task.id),
            "repeat_idx": repeat_idx,
            "status": TaskExecutionStatus.TASK_TIMEOUT.value,
            "traces": e.partial_traces,
            "config": {},
            "eval": None,
            "error": {
                "error_type": "TaskTimeoutError",
                "error_message": str(e),
                "elapsed": e.elapsed,
                "timeout": e.timeout,
            },
        }
    except AgentError as e:
        # ... existing error handling
        pass
```

---

## Implementation Phases

### Phase 0: Extract ComponentRegistry (Low Risk, Do First)

**Scope**: Extract registry logic from `Benchmark` into dedicated `ComponentRegistry` class.

**Files Modified**:

- `maseval/core/registry.py` (new): `ComponentRegistry` with thread-local storage
- `maseval/core/benchmark.py`: Replace inline registry with delegation to `ComponentRegistry`
- `maseval/core/__init__.py`: Export `ComponentRegistry`

**Effort**: ~1-2 days

**Breaking Changes**: None (public API unchanged, internal refactoring only)

**Why first**: This refactoring is needed for clean parallel execution. Doing it first:

- Isolates the thread-local complexity
- Makes subsequent phases simpler
- Can be tested and merged independently

### Phase 1: TaskProtocol & Timeout (Low Risk)

**Scope**: Add `TaskProtocol` dataclass, integrate cooperative timeout.

**Files Modified**:

- `maseval/core/task.py`: Add `TaskProtocol`, attach to `Task`
- `maseval/core/exceptions.py`: Add `TaskTimeoutError`
- `maseval/core/benchmark.py`: Add `TaskContext`, timeout checking in execution

**Effort**: ~2-3 days

**Breaking Changes**: None (new optional field with defaults)

### Phase 2: TaskQueue Abstraction (Medium Risk)

**Scope**: Extract task iteration into `TaskQueue`, maintain sequential default.

**Files Modified**:

- `maseval/core/queue.py` (new): `TaskQueue`, `SequentialQueue`, `PriorityQueue`
- `maseval/core/benchmark.py`: Refactor `run()` to use queue

**Effort**: ~3-4 days

**Breaking Changes**: None (signature changes are additive)

### Phase 3: Parallel Execution (Higher Risk)

**Scope**: Add `max_workers` parameter, thread-safe report collection, callback locking.

**Files Modified**:

- `maseval/core/benchmark.py`: Add `_run_parallel()`, `_invoke_callbacks()`, `_append_report_safe()`

**Effort**: ~4-5 days

**Breaking Changes**: None. MASEval handles all thread safety internally.

**Note**: Requires Phase 0 (ComponentRegistry) to be complete.

### Phase 4: AdaptiveQueue (Collaborator-Driven)

**Scope**: Implement `AdaptiveQueue` for IRT-based adaptive testing.

**Files Modified**:

- `maseval/core/queue.py`: Add `AdaptiveQueue` base or concrete implementation
- `maseval/core/callback.py`: Add `on_task_selected`, `on_queue_decision` (if needed)

**Effort**: ~3-4 days (depends on algorithm complexity)

**Breaking Changes**: None

**Note**: This phase will be driven by collaborator implementing their adaptive sampling paper. MASEval provides the `TaskQueue` interface; they implement the selection algorithm.

---

## Risks and Mitigations

### Risk 1: Thread Safety Bugs

**Mitigation**:

- Thread-local storage for per-task registries (already ephemeral per-repetition)
- Lock for shared report list
- Lock for callback invocations (users don't need to think about this)
- Default to `max_workers=1` for backwards compatibility
- Comprehensive tests with race condition detection

### Risk 2: Framework Incompatibility

**Mitigation**:

- Test with all supported frameworks (smolagents, langgraph, llamaindex)
- Document that user's `run_agents()` should not rely on shared mutable benchmark state
- All current adapters are stateless per-invocation (already safe)

### Risk 3: Timeout Incomplete Cleanup

**Mitigation**:

- Cooperative timeout with checkpoints (clean interruption points)
- Hard timeout as backstop—logs warning but continues gracefully
- Document that timed-out tasks may leave external resources (API connections) in undefined state
- Timeout is "best effort"—we cannot forcibly kill Python threads

### Risk 4: Callback Ordering in Parallel Mode

**Mitigation**:

- In parallel mode, `on_task_repeat_end` order is non-deterministic
- Document this behavior clearly
- Callbacks are still serialized (never concurrent), just out-of-order

### Risk 5: Memory Pressure with Many Workers

**Mitigation**:

- Default `max_workers=1`
- Document memory implications
- Consider `max_workers="auto"` that uses `os.cpu_count()`

---

## Summary

### Implementation Order

| Phase | Feature                      | Risk   | Effort   | Dependencies |
| ----- | ---------------------------- | ------ | -------- | ------------ |
| 0     | ComponentRegistry extraction | Low    | 1-2 days | None         |
| 1     | TaskProtocol & Timeout       | Low    | 2-3 days | None         |
| 2     | TaskQueue abstraction        | Medium | 3-4 days | None         |
| 3     | Parallel Execution           | Higher | 4-5 days | Phase 0      |
| 4     | AdaptiveQueue                | Medium | 3-4 days | Phase 2      |

### Feature Summary

| Feature             | Approach                                      | Breaking Changes |
| ------------------- | --------------------------------------------- | ---------------- |
| ComponentRegistry   | Extracted class with thread-local state       | None             |
| Parallel Processing | `ThreadPoolExecutor` with `max_workers` param | None             |
| Timeout Handling    | Cooperative checkpoints + hard backstop       | None             |
| TaskQueue           | Iterator abstraction with `on_task_complete`  | None             |
| Callback Safety     | MASEval serializes with internal lock         | None             |

### Key Design Decisions

1. **Extract ComponentRegistry**: Separate concerns. Registry manages thread-local component tracking. Benchmark orchestrates execution. Enables clean parallel implementation.

2. **Threading over asyncio**: No user code changes required. Works with all agent frameworks (including sync-only smolagents). Future-proof for Python's GIL removal.

3. **MASEval-managed callback safety**: All callback invocations are serialized with a lock. Users never need to think about thread safety in their callbacks.

4. **Cooperative timeout**: Cross-platform, works in threads, clean interruption at defined checkpoints. Hard timeout as backstop for misbehaving code (best-effort only—Python threads cannot be killed).

5. **AdaptiveQueue for collaborator**: The `TaskQueue` interface enables a collaborator to implement their adaptive sampling paper. MASEval provides the hooks; they implement the algorithm.

### What's NOT Changing

- **Public API**: All existing methods work unchanged
- **User-implemented methods**: `run_agents()`, `setup_environment()`, etc. stay sync
- **Callback interface**: Users write callbacks exactly as today
- **Default behavior**: `max_workers=1` maintains sequential execution

The unified design maintains **full backwards compatibility** while enabling:

- **Faster benchmarks** through parallelism
- **Resource-bounded execution** through timeouts
- **Intelligent task selection** through adaptive queues

All features share the same execution model refactor, making them natural to implement together.
