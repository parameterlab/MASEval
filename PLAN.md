# CAMEL-AI Integration Plan

## Overview

This document outlines the implementation plan for integrating CAMEL-AI as an agent framework adapter in MASEval.

## CAMEL-AI Ecosystem Overview

**CAMEL-AI** is an open-source, modular framework for building intelligent multi-agent systems. The ecosystem includes:

| Project | Description | Type |
|---------|-------------|------|
| **CAMEL** | Core multi-agent framework | Agent Framework |
| **OWL** | Optimized Workforce Learning - extends CAMEL's RolePlaying | Agent Framework (extends CAMEL) |
| **CRAB** | Cross-environment Agent Benchmark for cross-device tasks | Benchmark/Environment |

### Sources
- CAMEL Documentation: https://docs.camel-ai.org/
- CAMEL GitHub: https://github.com/camel-ai/camel
- OWL GitHub: https://github.com/camel-ai/owl
- OWL EnhancedRolePlaying: https://github.com/camel-ai/owl/blob/main/owl/utils/enhanced_role_playing.py
- CRAB Website: https://crab.camel-ai.org/

---

## Terminology Clarification

**Important**: The term "Task" is overloaded between MASEval and CAMEL-AI:

| Term | Definition | Scope |
|------|------------|-------|
| **maseval.Task** | A data point in a benchmark dataset. One `maseval.Task` = one evaluation problem (e.g., one GAIA question). | Evaluation level |
| **camelai.Task** | An internal work unit that agents decompose and solve. Many `camelai.Tasks` can exist within solving ONE `maseval.Task`. | Agent collaboration level |

These operate at **different levels** and do not conflict.

---

## Architectural Analysis: CAMEL vs MASEval

### Key Insight: Complementary Orchestration Layers

MASEval and CAMEL orchestrate at **different levels** - they complement rather than compete:

| Component | Orchestrates | Level | Scope |
|-----------|--------------|-------|-------|
| **MASEval Benchmark** | Evaluation lifecycle | Evaluation | Running many `maseval.Tasks`, collecting traces, metrics, repetitions |
| **MASEval `execution_loop`** | Agent-user turns | Interaction | Multi-turn conversation within ONE `maseval.Task` |
| **CAMEL RolePlaying** | Agent-agent turns | Interaction | Alternates between `assistant_agent` and `user_agent` |
| **CAMEL Workforce** | Agent collaboration | Collaboration | Task decomposition, assignment, retries within ONE problem |

**Key realization**: Workforce handles how agents collaborate to solve ONE `maseval.Task` by decomposing it into many `camelai.Tasks`. MASEval handles running MANY `maseval.Tasks` and collecting evaluation metrics. **They work together, not against each other.**

### Agent-to-Agent Evaluation

MASEval already supports agent-to-agent evaluation in multiple ways:

1. **Multi-agent frameworks without User**: LangGraph, AutoGen, and other frameworks have multiple agents communicating internally. MASEval wraps and evaluates the entire system. The `maseval.User` is optional - `setup_user()` can return `None`.

2. **LLM-simulated user**: The `maseval.User` with `UserLLMSimulator` IS an LLM-based agent simulating user behavior.

3. **Potential: Framework-specific agent as user**: With a `BaseUser` abstraction, any agent implementation could serve as the "user" side (see Option B).

### Component Deep Dive

#### RolePlaying
- Contains two `ChatAgent` instances as **public attributes**: `assistant_agent`, `user_agent`
- `step()` method alternates between agents, returns `Tuple[ChatAgentResponse, ChatAgentResponse]`
- Handles termination via `terminated` flags
- Each agent maintains its own memory via `ChatAgent.memory`
- **State worth tracing**: Configuration (task_prompt, roles), step count, termination reason
- **Does NOT maintain**: Aggregate metrics or conversation-level logging beyond individual agents

#### OWL (OwlRolePlaying)
```python
class OwlRolePlaying(RolePlaying):  # Inherits from CAMEL's RolePlaying
    self.assistant_agent: ChatAgent  # Accessible
    self.user_agent: ChatAgent       # Accessible
```
- Adds enhanced prompts and GAIA-specific formatting (`<analysis>`, `<final_answer>` tags)
- Same underlying architecture - agents are extractable
- No additional runtime state beyond RolePlaying

#### Workforce
Complex task orchestrator with rich state worth tracing:

| Attribute | Description | Tracing Value |
|-----------|-------------|---------------|
| `_children` | List of worker nodes (agents) | Worker roster |
| `_pending_tasks` / `_completed_tasks` | Task queues | Task lifecycle |
| `_task_dependencies` | Dependency graph | How problem was decomposed |
| `_assignees` | Worker assignments | Who did what |
| `_snapshots` | Checkpoints | Execution history |
| `WorkforceMetrics` | Lifecycle events | Task creation, assignment, completion, failure |
| Recovery strategies | RETRY, REPLAN, REASSIGN, DECOMPOSE, CREATE_WORKER | Debugging agent behavior |

#### Memory
- `agent.memory.get_context()` returns `(List[OpenAIMessage], int)` - messages + token count
- Already traced in our `CamelAgentAdapter.get_messages()` implementation

---

## Phase 1: ChatAgent Adapter

### Status: IMPLEMENTED (Testing Incomplete)

Basic CAMEL-AI integration supporting the core `ChatAgent` class. Implementation complete, but not yet added to contract tests.

### Completed Items

| Task | Status | Notes |
|------|--------|-------|
| Create `maseval/interface/agents/camel.py` | Done | `CamelAgentAdapter` and `CamelUser` |
| Update `maseval/interface/agents/__init__.py` | Done | Conditional import added |
| Add `camel` optional dependency to `pyproject.toml` | Done | `camel-ai>=0.2.0` |
| Add `camel` pytest marker | Done | In `pyproject.toml` |
| Create integration tests | Done | 17 tests, all passing |
| Create documentation | Done | `docs/interface/agents/camel.md` |
| Update mkdocs.yml navigation | Done | Added to Interface > Agents |
| Update CHANGELOG.md | Done | Under Unreleased |
| Linting (`ruff format`, `ruff check`) | Done | Passes |
| Type checking (`ty check`) | Done | Passes |
| Integration tests (`pytest -m camel`) | Done | 17/17 passing |
| **Add to contract tests** | TODO | See "Outstanding (Phase 1 - Testing)" below |
| **Improve test coverage** | TODO | `_run_agent()`, error paths, multi-turn |

### Implementation Details

**CamelAgentAdapter**:
- Wraps CAMEL `ChatAgent` instances
- Uses `agent.step(user_msg)` for execution
- Fetches messages from `agent.memory.get_context()` (source of truth pattern)
- Captures response metadata (`terminated`, `info`) in traces
- Extracts tool and model configuration in `gather_config()`

**CamelUser**:
- Extends base `User` class
- `get_tool()` returns CAMEL `FunctionTool` wrapping `simulate_response`

### Design Decisions Made

1. **Memory as Source of Truth**: Using `agent.memory.get_context()` following AGENTS.md Pattern 1
2. **Minimal Message Conversion**: CAMEL already uses OpenAI-compatible format
3. **Step-based Execution**: Using `step()` method (synchronous, single-turn)
4. **No `camel_optional.py`**: Not needed since tool creation is simple (inline import)

---

## Phase 2: Multi-Agent Integration Options

### Option A: Agent Extraction (Documentation Only)

**Approach**: Document how users can extract individual `ChatAgent` instances from RolePlaying/OWL/Workforce and wrap them with existing `CamelAgentAdapter`.

**Pattern for RolePlaying/OWL**:
```python
from camel.societies import RolePlaying
from maseval.interface.agents.camel import CamelAgentAdapter

class MyBenchmark(Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        # Create RolePlaying
        role_playing = RolePlaying(
            assistant_role_name="Assistant",
            user_role_name="User",
            task_prompt=task.query,
        )

        # Extract and wrap the assistant agent
        assistant_adapter = CamelAgentAdapter(
            role_playing.assistant_agent,
            name="camel_assistant"
        )

        return [assistant_adapter], {"assistant": assistant_adapter}
```

**Pattern for Workforce**:
```python
from camel.societies.workforce import Workforce
from maseval.interface.agents.camel import CamelAgentAdapter

class MyBenchmark(Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        # Create Workforce
        workforce = Workforce(...)
        self._workforce = workforce  # Store for run_agents

        # Extract and wrap individual workers for tracing
        worker_adapters = {}
        for worker in workforce._children:
            adapter = CamelAgentAdapter(worker.agent, name=worker.name)
            worker_adapters[worker.name] = adapter

        # Return empty agents_to_run (we'll run Workforce directly)
        return [], worker_adapters

    def run_agents(self, agents, task, environment, query):
        # Run Workforce (handles internal orchestration)
        result = self._workforce.process(query)
        return result
```

**Pros**:
- No new code needed in MASEval
- Clean separation: MASEval evaluates, CAMEL orchestrates agent collaboration
- Works for RolePlaying, OWL, and Workforce

**Cons**:
- Users must understand CAMEL internals
- Workforce orchestration state (task decomposition, assignments) not automatically traced

**Effort**: Low (documentation only)

---

### Option B: BaseUser Abstraction

**Motivation**: The current `maseval.User` forces LLM simulation via `UserLLMSimulator`. This is too restrictive for:
- Using a smolagents agent as user
- Using CAMEL's `user_agent` from RolePlaying
- Connecting user to an MCP server
- Scripted/deterministic users for testing
- Human-in-the-loop scenarios

**Approach**: Refactor `maseval.User` into an abstract `BaseUser` and concrete implementations.

**Core changes** (in `maseval/core/user.py`):
```python
class BaseUser(ABC, TraceableMixin, ConfigurableMixin):
    """Abstract base class defining the user interface."""

    @abstractmethod
    def simulate_response(self, question: str) -> str:
        """Generate a response to the agent's question."""
        ...

    @abstractmethod
    def get_initial_query(self) -> str:
        """Return the initial query to start the conversation."""
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """Check if the user interaction should terminate."""
        ...

    @abstractmethod
    def get_tool(self):
        """Return a framework-compatible tool for agent interaction."""
        ...

class SimulatedUser(BaseUser):
    """LLM-simulated user (current User implementation)."""
    def __init__(self, model: ModelAdapter, ...):
        self._llm_simulator = UserLLMSimulator(model, ...)
    # ... current implementation
```

**Interface additions** (in `maseval/interface/agents/camel.py`):
```python
class CamelAgentUser(BaseUser):
    """User backed by a CAMEL ChatAgent."""

    def __init__(self, user_agent: ChatAgent, initial_query: str, max_turns: int = 10):
        self.user_agent = user_agent
        self._initial_query = initial_query
        self._turn_count = 0
        self._max_turns = max_turns

    def simulate_response(self, question: str) -> str:
        self._turn_count += 1
        response = self.user_agent.step(question)
        return response.msgs[0].content

    def get_initial_query(self) -> str:
        return self._initial_query

    def is_done(self) -> bool:
        return self._turn_count >= self._max_turns

    def get_tool(self):
        from camel.toolkits import FunctionTool
        return FunctionTool(self.simulate_response)
```

**Benefits**:
- Follows MASEval's "Abstract Base Classes" principle
- Enables any agent implementation as user (CAMEL, smolagents, LangGraph, MCP, etc.)
- Library-wide improvement, not CAMEL-specific

**Effort**: Medium (core refactor, but library is early-release)

---

### Option C: Custom execution_loop Override

**Context**: The `execution_loop` method is **designed to be overridden**. From the docstring:

> *"Override this method in your benchmark subclass to implement custom interaction patterns (e.g., agent-initiated conversations, different termination conditions, or specialized query routing)."*

**Approach**: Users can override `execution_loop` to use RolePlaying's `step()` semantics:

```python
class CamelRolePlayingBenchmark(Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        self._role_playing = RolePlaying(
            assistant_role_name="Assistant",
            user_role_name="User",
            task_prompt=task.query,
        )

        # Wrap both agents for tracing
        assistant = CamelAgentAdapter(self._role_playing.assistant_agent, "assistant")
        user_agent = CamelAgentAdapter(self._role_playing.user_agent, "user_agent")

        return [assistant], {"assistant": assistant, "user_agent": user_agent}

    def execution_loop(self, agents, task, environment, user):
        """Use RolePlaying's step() instead of default agent-user loop."""
        self._role_playing.init_chat()

        final_answer = None
        for _ in range(self.max_invocations):
            assistant_response, user_response = self._role_playing.step()

            if assistant_response.msgs:
                final_answer = assistant_response.msgs[-1].content

            if assistant_response.terminated or user_response.terminated:
                break

        return final_answer
```

**Use case**: When you want RolePlaying's exact interaction semantics (CAMEL's `user_agent` drives the conversation) rather than MASEval's `maseval.User`.

**Pros**:
- Uses existing extension point
- Full control over interaction pattern
- Both agents traced via adapters

**Cons**:
- More code for users to write
- Bypasses `maseval.User` entirely

**Effort**: Low (documentation + example)

---

### Option D: Trace Wrappers for Orchestration State

**Motivation**: RolePlaying and Workforce maintain orchestration state that individual agent traces don't capture. Users may want to include this state in MASEval traces for debugging and analysis.

**Approach**: Provide lightweight wrappers implementing `TraceableMixin` that users can register.

**RolePlaying Tracer**:
```python
class CamelRolePlayingTracer(TraceableMixin, ConfigurableMixin):
    """Collects orchestration traces from CAMEL RolePlaying."""

    def __init__(self, role_playing: RolePlaying, name: str = "role_playing"):
        self.role_playing = role_playing
        self.name = name
        self._step_count = 0
        self._termination_reason: Optional[str] = None

    def record_step(self, assistant_response, user_response):
        """Call this after each step() to track progress."""
        self._step_count += 1
        if assistant_response.terminated:
            self._termination_reason = "assistant_terminated"
        elif user_response.terminated:
            self._termination_reason = "user_terminated"

    def gather_traces(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "role_playing_orchestration",
            "step_count": self._step_count,
            "termination_reason": self._termination_reason,
        }

    def gather_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_prompt": self.role_playing.task_prompt,
            "assistant_role": getattr(self.role_playing, 'assistant_role_name', None),
            "user_role": getattr(self.role_playing, 'user_role_name', None),
        }
```

**Workforce Tracer**:
```python
class CamelWorkforceTracer(TraceableMixin, ConfigurableMixin):
    """Collects orchestration traces from CAMEL Workforce."""

    def __init__(self, workforce: Workforce, name: str = "workforce"):
        self.workforce = workforce
        self.name = name

    def gather_traces(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "workforce_orchestration",
            "task_decomposition": self._extract_task_graph(),
            "worker_assignments": dict(getattr(self.workforce, '_assignees', {})),
            "completed_tasks": self._extract_completed_tasks(),
            "pending_tasks": len(getattr(self.workforce, '_pending_tasks', [])),
        }

    def _extract_task_graph(self) -> Dict[str, Any]:
        deps = getattr(self.workforce, '_task_dependencies', {})
        return {task_id: list(dep_ids) for task_id, dep_ids in deps.items()}

    def _extract_completed_tasks(self) -> List[Dict[str, Any]]:
        completed = getattr(self.workforce, '_completed_tasks', [])
        return [{"id": t.id, "content": t.content} for t in completed]

    def gather_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": str(getattr(self.workforce, 'mode', 'unknown')),
            "workers": [w.name for w in getattr(self.workforce, '_children', [])],
        }
```

**Usage**:
```python
class MyBenchmark(Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        workforce = Workforce(...)

        # Create tracer and register it
        workforce_tracer = CamelWorkforceTracer(workforce)
        self.register(workforce_tracer)  # Traces go to "other" category

        # Wrap individual workers
        worker_adapters = {...}

        return [], worker_adapters
```

**What gets traced**:
- Individual agents: Full message history via `CamelAgentAdapter`
- Orchestration: Task decomposition, assignments, lifecycle via tracer

**Pros**:
- Uses existing `TraceableMixin` / `register()` pattern
- Clear separation: agents vs orchestration traces
- Optional - users only add if they need orchestration visibility

**Cons**:
- Users must manually call `record_step()` for RolePlaying
- Workforce internal state accessed via private attributes (may break)

**Effort**: Medium

---

## Integration Summary

| CAMEL Component | Recommended Integration | New Code Needed |
|-----------------|------------------------|-----------------|
| **ChatAgent** | `CamelAgentAdapter` | Done |
| **RolePlaying** | Option A (extract agents) + Option C (custom loop) or Option D (tracer) | Documentation + optional tracer |
| **OWL** | Same as RolePlaying | Same as RolePlaying |
| **Workforce** | Option A (extract workers) + Option D (tracer) | Documentation + optional tracer |
| **Society** | Extract agents | Documentation only |
| **Memory** | Already traced via `get_context()` | None |

---

## Recommendations

### Phase 2a: Documentation + Tracers (Recommended Next Step)

1. **Document extraction patterns** (Option A):
   - How to use RolePlaying with MASEval
   - How to use Workforce with MASEval
   - How to override `execution_loop` for custom interaction (Option C)

2. **Implement optional tracers** (Option D):
   - `CamelRolePlayingTracer` - lightweight, captures step count and termination
   - `CamelWorkforceTracer` - captures task decomposition, assignments, lifecycle

### Phase 2b: BaseUser Abstraction (Future)

If there's demand for flexible user implementations:

1. **Core refactor**: Split `User` → `BaseUser` + `SimulatedUser`
2. **Interface additions**: `CamelAgentUser`, potentially `SmolAgentUser`, `MCPUser`, etc.

This is a library-wide improvement that benefits all frameworks, not just CAMEL.

### Decision Framework

| If you need... | Recommended approach |
|----------------|---------------------|
| Evaluate CAMEL ChatAgents | `CamelAgentAdapter` (Phase 1) |
| Use RolePlaying with MASEval's user simulation | Extract `assistant_agent`, wrap with adapter |
| Use RolePlaying's `user_agent` | Override `execution_loop` (Option C) or wait for BaseUser (Option B) |
| Use Workforce with MASEval | Extract workers, optionally add `CamelWorkforceTracer` |
| Trace Workforce task decomposition | `CamelWorkforceTracer` (Option D) |

---

## Open Questions

1. **Tracer implementation location**: Should tracers go in `maseval/interface/agents/camel.py` or a separate `maseval/interface/tracers/` directory?

2. **Workforce private attributes**: The tracer accesses `_children`, `_assignees`, `_task_dependencies`, etc. Should we document this as "may break with CAMEL updates" or request stable APIs from CAMEL-AI?

3. **BaseUser timing**: Should BaseUser be done as part of CAMEL integration, or as a separate MASEval improvement?

4. **Trace category**: Should orchestration traces go to a dedicated category (e.g., `orchestration`) or use the existing `other` category in `collect_all_traces()`?

---

## Files Reference

### Completed (Phase 1)
- `maseval/interface/agents/camel.py` - Main adapter (`CamelAgentAdapter`, `CamelUser`)
- `maseval/interface/agents/__init__.py` - Conditional import
- `pyproject.toml` - Dependency and marker
- `tests/test_interface/test_agent_integration/test_camel_integration.py` - Integration tests (17 tests)
- `docs/interface/agents/camel.md` - Documentation
- `mkdocs.yml` - Navigation
- `CHANGELOG.md` - Entry

### Outstanding (Phase 1 - Testing)

**Contract Tests Gap**: CAMEL is NOT included in contract tests.

The contract tests (`tests/test_contract/test_agent_adapter_contract.py`) validate that all `AgentAdapter` implementations honor the same interface. Currently tested frameworks:
- `dummy`
- `smolagents`
- `langgraph`
- `llamaindex`
- `camel` **MISSING**

**Required changes to add CAMEL to contract tests:**

1. Add `"camel"` to the `@pytest.mark.parametrize("framework", [...])` decorator
2. Add CAMEL case to `create_agent_for_framework()`:
   ```python
   elif framework == "camel":
       pytest.importorskip("camel")
       from camel.agents import ChatAgent
       from camel.messages import BaseMessage

       # Create a mock ChatAgent for testing
       class MockCamelAgent:
           """Mock CAMEL ChatAgent for contract testing."""
           def __init__(self, response: str = "Test response"):
               self.response = response
               self.memory = MockMemory()

           def step(self, user_msg):
               # Return mock ChatAgentResponse
               ...

       return MockCamelAgent(response=mock_llm.responses[0])
   ```

3. Add CAMEL case to `create_adapter_for_framework()`:
   ```python
   elif framework == "camel":
       pytest.importorskip("camel")
       from maseval.interface.agents.camel import CamelAgentAdapter
       return CamelAgentAdapter(agent, "test_agent", callbacks=callbacks)
   ```

### Potential: Phase 2a (Tracers)
- `maseval/interface/agents/camel.py` - Add `CamelRolePlayingTracer`, `CamelWorkforceTracer`
- `docs/interface/agents/camel.md` - Add extraction patterns and tracer usage
- `tests/test_interface/test_agent_integration/test_camel_integration.py` - Add tracer tests

### Potential: Phase 2b (BaseUser)

**Core changes:**
- `maseval/core/user.py` - Split `User` → `BaseUser` + `SimulatedUser`
- `maseval/core/__init__.py` - Export `BaseUser`, `SimulatedUser`
- `maseval/core/benchmark.py` - Update type hints to use `BaseUser`
- `tests/test_core/test_user.py` - Update tests for new structure

**Interface additions:**
- `maseval/interface/agents/camel.py` - Add `CamelAgentUser(BaseUser)`
- `maseval/interface/agents/smolagents.py` - Update `SmolAgentUser` to extend `SimulatedUser`
- `maseval/interface/agents/langgraph.py` - Update `LangGraphUser` to extend `SimulatedUser`
- `maseval/interface/agents/llamaindex.py` - Update `LlamaIndexUser` to extend `SimulatedUser`
