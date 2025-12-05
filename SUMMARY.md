# Implementation Summary: Parallel Task Execution Engine

This document summarizes the implementation of parallel task execution, timeout handling, and task queue abstraction for MASEval, as specified in PLAN.md.

## Phase 0: ComponentRegistry Extraction

**New File:** `maseval/core/registry.py`

- Created `ComponentRegistry` class that manages component registration for tracing and configuration collection
- Uses `threading.local()` for thread-local storage, enabling parallel task execution without cross-contamination between threads
- Each thread gets isolated registry state: `_trace_registry`, `_component_id_map`, `_config_registry`, `_config_component_id_map`
- Methods: `register()`, `clear()`, `collect_traces()`, `collect_configs()`
- Refactored `Benchmark` class to delegate all registry operations to `self._registry: ComponentRegistry`

## Phase 1: TaskProtocol & Timeout Infrastructure

**Modified:** `maseval/core/task.py`

- Added `TimeoutAction` enum with values `SKIP`, `RETRY`, `RAISE` for configurable timeout behavior
- Added `TaskProtocol` dataclass with fields:
  - `timeout_seconds: Optional[float]` - per-task timeout limit
  - `timeout_action: TimeoutAction` - what to do on timeout (default: SKIP)
  - `max_retries: int` - retry count for failed tasks (default: 0)
  - `priority: int` - scheduling priority (default: 0, higher = more important)
  - `tags: Dict[str, Any]` - arbitrary metadata for filtering/grouping
- Added `protocol: TaskProtocol` field to `Task` dataclass

**New File:** `maseval/core/context.py`

- Created `TaskContext` class for cooperative timeout checking
- Properties: `elapsed` (time since start), `remaining` (time until deadline), `is_expired` (bool)
- Method: `check_timeout()` raises `TaskTimeoutError` if deadline exceeded
- Designed for checkpoint-based timeout checking in user code

**Modified:** `maseval/core/exceptions.py`

- Added `TaskTimeoutError(MASEvalError)` with attributes:
  - `elapsed: float` - how long the task ran
  - `timeout: float` - the configured timeout limit
  - `partial_traces: Optional[Dict]` - any traces collected before timeout

**Modified:** `maseval/core/benchmark.py`

- Added `TASK_TIMEOUT` to `TaskExecutionStatus` enum

## Phase 2: TaskQueue Abstraction

**New File:** `maseval/core/queue.py`

- Created `TaskQueue` abstract base class with iterator interface (`__iter__`, `__next__`)
- Supports both `Task` and `TaskCollection` inputs, with automatic expansion
- Handles `n_task_repeats` by yielding `(task, repeat_idx)` tuples

**Implementations:**

1. `SequentialQueue` - Simple FIFO ordering, iterates tasks in input order
2. `PriorityQueue` - Uses `TaskProtocol.priority` for scheduling (higher priority first)
3. `AdaptiveQueue` - Placeholder for future feedback-based scheduling (currently falls back to sequential)

## Phase 3: Parallel Execution

**Modified:** `maseval/core/benchmark.py`

- Added `max_workers: int = 1` parameter to `Benchmark.run()` for controlling parallelism
- Added `queue: Optional[TaskQueue] = None` parameter for custom scheduling (defaults to `SequentialQueue`)
- Added thread-safety mechanisms:
  - `self._reports_lock: threading.Lock` for safe report collection from multiple threads
  - `self._callback_lock: threading.Lock` for serialized callback invocation
- New methods:
  - `_invoke_callbacks(method_name, *args, **kwargs)` - thread-safe callback invocation
  - `_append_report_safe(report)` - thread-safe report collection
  - `_execute_task_repetition(task, repeat_idx, context)` - single task execution with timeout support
  - `_run_sequential(queue)` - sequential execution (backward compatible)
  - `_run_parallel(queue, max_workers)` - parallel execution using `ThreadPoolExecutor`

**Backward Compatibility:**

- `max_workers=1` (default) uses `_run_sequential()`, preserving existing behavior
- `max_workers>1` uses `_run_parallel()` with thread pool

## Phase 4: AdaptiveQueue (Placeholder)

- `AdaptiveQueue` class created as placeholder for collaborator implementation
- Intended for feedback-based scheduling that reorders remaining tasks based on execution results
- Currently falls back to sequential iteration

## Updated Exports

**Modified:** `maseval/__init__.py`

New public exports:

- `TaskProtocol`, `TimeoutAction` - task execution configuration
- `ComponentRegistry` - thread-safe component registration
- `TaskContext` - timeout checking context
- `TaskQueue`, `SequentialQueue`, `PriorityQueue`, `AdaptiveQueue` - scheduling abstractions
- `TaskTimeoutError` - timeout exception

## Test Updates

- Updated 2 test files that accessed internal registry attributes (`_trace_registry`, `_component_id_map`, `_config_registry`)
- Changed to access through `benchmark._registry._trace_registry` pattern
- All 666 tests pass
