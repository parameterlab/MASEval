# MARBLE Integration Strategy for MASEval

## Executive Summary

This document proposes the integration architecture for bringing MARBLE (Multi-Agent Coordination Backbone with LLM Engine) and its MultiAgentBench benchmark suite into MASEval.

**Key Architecture**: A dual-purpose design that enables:
1. **Exact MARBLE reproduction** via `MarbleMultiAgentBenchBenchmark` (for scientific validation)
2. **Fair framework comparison** via abstract `MultiAgentBenchBenchmark` base (users implement with their own frameworks)

This enables critical research questions: **"Which multi-agent framework performs best on the same tasks?"**

**Key Finding**: Individual MARBLE agents CAN be extracted and wrapped in separate AgentAdapters, providing per-agent trace visibility while preserving MARBLE's coordination logic.

**License**: ✅ MARBLE's MIT license explicitly permits vendoring/usage with attribution.

---

## Background

### What is MARBLE?

MARBLE is a multi-agent coordination framework that:
- Orchestrates multiple LLM-based agents using an **Engine** + **AgentGraph** architecture
- Supports various coordination modes (star, chain, tree, graph)
- Provides **inter-agent communication** via direct message passing (stored in `agent.msg_box`)
- Includes domain-specific environments (Coding, Database, Minecraft, Research, Bargaining, Web, WorldSimulation)
- Uses configuration for agent relationships and task specifications

### What is MultiAgentBench?

MultiAgentBench is MARBLE's benchmark suite featuring:
- **7 domains**: Coding, Database, Minecraft, Research, Bargaining, Web, WorldSimulation
- **JSONL task format** with rich metadata (agent relationships, coordination modes, evaluation metrics)
- **500+ tasks** testing collaboration and competition scenarios
- Original paper: "MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents" (arXiv:2503.01935)

### MASEval Integration Goals

1. **Reproduce MARBLE's published results** - Exact reproduction via MarbleMultiAgentBenchBenchmark
2. **Enable multi-agent framework comparison** - Abstract base allows users to implement with any framework
3. **Maintain MASEval's framework-agnostic design** - No hard dependencies on agent frameworks
4. **Reuse MASEval infrastructure** - Callbacks, tracing, parallelization, error handling
5. **Per-agent trace visibility** - Wrap individual agents, not just the engine

---

## Integration Architecture

### Component Structure

```
maseval/benchmark/multiagentbench/
├── __init__.py
├── README.md                              # Setup instructions
├── PROVENANCE.md                          # Track upstream version, license
├── .gitignore                             # Ignore marble/ directory
│
├── multiagentbench.py                     # Core classes:
│   ├── MultiAgentBenchBenchmark           #   - Abstract base (framework-agnostic)
│   ├── MarbleMultiAgentBenchBenchmark     #   - MARBLE reproduction
│   └── MultiAgentBenchEvaluator           #   - Wraps MARBLE metrics
│
├── environment.py                         # MultiAgentBenchEnvironment
├── data_loader.py                         # load_tasks(), configure_model_ids()
├── adapters/
│   └── marble_adapter.py                  # MarbleAgentAdapter (per-agent)
│
└── marble/                                # ← Vendored MARBLE (gitignored)
    ├── LICENSE                            #   MIT license preserved
    ├── agent/
    ├── engine/
    ├── environments/
    ├── evaluator/
    └── ...                                #   Full MARBLE source
```

### Class Hierarchy

```python
# Abstract base - provides task/eval infrastructure for ANY framework
class MultiAgentBenchBenchmark(Benchmark):
    """Framework-agnostic base for MultiAgentBench tasks.

    Users implement setup_agents() with their chosen framework.
    """
    def setup_environment(...)  → MultiAgentBenchEnvironment  # Shared
    def setup_evaluators(...)   → MultiAgentBenchEvaluator    # Shared
    def setup_agents(...)       → ABSTRACT                    # User implements

# MARBLE reproduction - wraps individual agents for trace visibility
class MarbleMultiAgentBenchBenchmark(MultiAgentBenchBenchmark):
    """Exact MARBLE reproduction for scientific validation."""
    def setup_agents(...):
        # Create MARBLE agents from task config
        # Wrap EACH agent in MarbleAgentAdapter
        # Return list of adapters to MASEval
```

---

## Agent Wrapping Strategy

### Per-Agent Adapters (Preferred Pattern)

MARBLE agents CAN be extracted and wrapped individually. This provides:
- Per-agent trace visibility (matches tau2/macs pattern)
- Better error attribution
- Callback hooks fire per-agent
- Cleaner debugging

**Requirements for independent agent execution:**
1. Each agent needs an `AgentGraph` reference (even if minimal)
2. Communicating agents must share the same `AgentGraph`
3. Each agent needs an environment reference
4. Call `agent.act(task)` directly

### How It Works

```python
class MarbleAgentAdapter(AgentAdapter):
    """Wraps a single MARBLE BaseAgent."""

    def __init__(
        self,
        agent: "BaseAgent",
        name: str,
        callbacks: Optional[List[Any]] = None,
    ):
        super().__init__(agent, name, callbacks)
        self._agent = agent

    def _run_agent(self, query: str) -> str:
        """Execute MARBLE agent's act() method."""
        result, communication = self._agent.act(query)
        return result

    def get_messages(self) -> MessageHistory:
        """Extract messages from agent's msg_box."""
        return self._extract_message_history()

    def _extract_message_history(self) -> MessageHistory:
        """Extract messages from MARBLE's msg_box structure.

        MARBLE stores messages in agent.msg_box:
            msg_box[session_id][other_agent_id] = List[(direction, message)]
            direction: 0 = FORWARD_TO (sent), 1 = RECV_FROM (received)
        """
        messages = []
        for session_id, conversations in self._agent.msg_box.items():
            for other_agent_id, msg_list in conversations.items():
                for direction, content in msg_list:
                    messages.append({
                        "role": "agent" if direction == 0 else "other",
                        "content": str(content),
                        "agent_id": self._agent.agent_id,
                        "other_agent_id": other_agent_id,
                        "direction": "sent" if direction == 0 else "received",
                        "session_id": session_id,
                    })
        return MessageHistory(messages)

    def gather_traces(self) -> Dict[str, Any]:
        """Gather MARBLE-specific execution data."""
        return {
            **super().gather_traces(),
            "agent_id": self._agent.agent_id,
            "agent_type": getattr(self._agent, "agent_type", "BaseAgent"),
            "profile": getattr(self._agent, "profile", ""),
            "task_history": getattr(self._agent, "task_history", []),
            "relationships": self._extract_relationships(),
        }

    def _extract_relationships(self) -> List[Dict[str, str]]:
        """Extract this agent's relationships from the graph."""
        relationships = []
        if self._agent.agent_graph:
            for src, dst, rel_type in self._agent.agent_graph.relationships:
                if src == self._agent.agent_id or dst == self._agent.agent_id:
                    relationships.append({
                        "source": src,
                        "target": dst,
                        "type": rel_type,
                    })
        return relationships
```

### Coordination Without Engine.start()

Instead of calling `engine.start()`, we implement coordination in MASEval's `run_agents()`:

```python
def run_agents(
    self,
    agents: Sequence[AgentAdapter],
    task: Task,
    environment: MultiAgentBenchEnvironment,
    query: str = "",
) -> Any:
    """Execute agents according to coordination mode."""
    coordinate_mode = task.environment_data.get("coordinate_mode", "star")

    if coordinate_mode == "star":
        return self._star_coordinate(agents, task, query)
    elif coordinate_mode == "chain":
        return self._chain_coordinate(agents, task, query)
    elif coordinate_mode == "tree":
        return self._tree_coordinate(agents, task, query)
    elif coordinate_mode == "graph":
        return self._graph_coordinate(agents, task, query)
    else:
        raise ValueError(f"Unknown coordination mode: {coordinate_mode}")

def _star_coordinate(
    self,
    agents: Sequence[AgentAdapter],
    task: Task,
    query: str,
) -> Any:
    """Star coordination: central planner assigns tasks to agents."""
    max_iterations = task.environment_data.get("max_iterations", 10)
    results = {}

    for iteration in range(max_iterations):
        # Assign tasks (simplified - full impl uses EnginePlanner)
        task_assignments = self._assign_tasks(agents, task, results)

        if not task_assignments:
            break

        # Execute each agent with its assigned task
        for agent, agent_task in task_assignments.items():
            result = agent.run(agent_task)
            results[agent.name] = result

        # Check termination
        if self._should_terminate(results, task):
            break

    return self._aggregate_results(results)
```

---

## Critical MARBLE API Details

### Message History Storage

**MARBLE stores messages in `agent.msg_box`, NOT SharedMemory:**

```python
# BaseAgent structure (base_agent.py)
self.msg_box: DefaultDict[str, DefaultDict[str, List[Tuple[int, str]]]]
# Structure: msg_box[session_id][other_agent_id] = [(direction, message), ...]
# Direction: 0 = FORWARD_TO (sent), 1 = RECV_FROM (received)

# Serialize via (note: typo in MARBLE)
serialized = agent.seralize_message(session_id="")
```

### SharedMemory is NOT Shared

Despite the name, each MARBLE agent creates its own `SharedMemory` instance:

```python
# In BaseAgent.__init__() (base_agent.py:72-73)
self.memory = BaseMemory()        # Per-agent memory
self.shared_memory = SharedMemory()  # Also per-agent, NOT shared!
```

**Do NOT rely on SharedMemory for inter-agent state.** Use `msg_box` or environment state instead.

### AgentGraph is Required

`agent.act()` requires an AgentGraph reference:

```python
# In BaseAgent.act() (base_agent.py:145-147)
assert self.agent_graph is not None, \
    "Agent graph is not set. Please set the agent graph using set_agent_graph method first."
```

**Solution:** Create AgentGraph during setup and set it on all agents:

```python
def setup_agents(self, agent_data, environment, task, user):
    # Create agents
    agents = self._create_agents(task)

    # Create and configure AgentGraph
    from .marble.graph.agent_graph import AgentGraph
    graph = AgentGraph(agents, self._build_graph_config(task))

    # Set graph reference on each agent (REQUIRED)
    for agent in agents:
        agent.set_agent_graph(graph)

    # Wrap in adapters
    adapters = [MarbleAgentAdapter(agent, agent.agent_id) for agent in agents]
    agents_dict = {a.name: a for a in adapters}

    return adapters, agents_dict
```

### Known MARBLE Bug: chain_coordinate()

**Bug:** `engine.py:702` calls `self.graph.get_agent_profiles_linked()` which does not exist in AgentGraph.

**Impact:** Chain coordination mode will crash.

**Workaround:** Either:
1. Avoid chain mode tasks initially
2. Patch MARBLE locally (add the missing method)
3. Implement chain coordination in MASEval's `run_agents()`

---

## Environment Integration

### Domain-Specific Requirements

| Domain | External Dependencies | Initial Support |
|--------|----------------------|-----------------|
| Research | None | ✅ Yes |
| Bargaining | None | ✅ Yes |
| Coding | Filesystem access | ✅ Yes |
| Web | Network access | ✅ Yes |
| WorldSimulation | None | ✅ Yes |
| Database | Docker + PostgreSQL | ⚠️ Optional |
| Minecraft | External game server | ❌ Deferred |

**Strategy:** Start with domains that don't require external services. Add infrastructure-heavy domains as optional extras with clear skip logic.

### MultiAgentBenchEnvironment

```python
class MultiAgentBenchEnvironment(Environment):
    """Wraps MARBLE environment instances."""

    # Domains that require external infrastructure
    INFRASTRUCTURE_DOMAINS = {"database", "minecraft"}

    def __init__(
        self,
        task_data: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
    ):
        self.domain = task_data.get("scenario", "")
        self._marble_env: Optional["BaseEnvironment"] = None
        super().__init__(task_data, callbacks)

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state and create MARBLE environment."""
        domain = task_data.get("scenario", "")
        env_config = task_data.get("environment", {})

        # Check infrastructure requirements
        if domain in self.INFRASTRUCTURE_DOMAINS:
            if not self._check_infrastructure(domain):
                raise EnvironmentError(
                    f"Domain '{domain}' requires external infrastructure. "
                    f"See README.md for setup instructions.",
                    component="MultiAgentBenchEnvironment",
                )

        # Create MARBLE environment
        self._marble_env = self._create_marble_environment(domain, env_config)

        return {
            "domain": domain,
            "env_config": env_config,
            "marble_env_type": type(self._marble_env).__name__,
        }

    def _check_infrastructure(self, domain: str) -> bool:
        """Check if required infrastructure is available."""
        if domain == "database":
            # Check Docker availability
            import shutil
            return shutil.which("docker") is not None
        elif domain == "minecraft":
            # Minecraft requires external server - always fail for now
            return False
        return True

    def _create_marble_environment(
        self,
        domain: str,
        env_config: Dict[str, Any],
    ) -> "BaseEnvironment":
        """Create the appropriate MARBLE environment."""
        from .marble.environments.base_env import BaseEnvironment

        # Import domain-specific environments
        env_classes = {
            "coding": "marble.environments.coding_env.CodingEnvironment",
            "database": "marble.environments.db_env.DBEnvironment",
            "research": "marble.environments.research_env.ResearchEnvironment",
            "bargaining": "marble.environments.bargaining_env.BargainingEnvironment",
            "web": "marble.environments.web_env.WebEnvironment",
            "worldsimulation": "marble.environments.world_env.WorldSimulationEnvironment",
        }

        env_class_path = env_classes.get(domain.lower())
        if not env_class_path:
            # Fallback to base environment
            return BaseEnvironment(env_config)

        # Dynamic import
        module_path, class_name = env_class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        env_class = getattr(module, class_name)
        return env_class(env_config)

    def create_tools(self) -> Dict[str, Any]:
        """Extract tools from MARBLE environment for tracing.

        MARBLE environments expose tools via action_handler_descriptions.
        We wrap these for MASEval tracing.
        """
        if not self._marble_env:
            return {}

        tools = {}
        for action_name in self._marble_env._action_handlers:
            handler = self._marble_env._action_handlers[action_name]
            # Wrap handler for tracing
            tools[action_name] = self._wrap_tool_for_tracing(action_name, handler)
        return tools

    def _wrap_tool_for_tracing(
        self,
        name: str,
        handler: Callable,
    ) -> Callable:
        """Wrap a MARBLE action handler for MASEval tracing."""
        tool_history = ToolInvocationHistory()

        def traced_handler(**kwargs) -> Any:
            try:
                result = handler(**kwargs)
                tool_history.add_invocation(
                    inputs=kwargs,
                    outputs=result,
                    status="success",
                )
                return result
            except Exception as e:
                tool_history.add_invocation(
                    inputs=kwargs,
                    outputs=str(e),
                    status="error",
                )
                raise

        # Attach history for trace collection
        traced_handler._history = tool_history
        traced_handler._original_name = name
        return traced_handler

    def get_tool(self, name: str) -> Optional[Any]:
        """Get a specific tool by name."""
        return self.tools.get(name)

    def gather_traces(self) -> Dict[str, Any]:
        """Gather traces including tool invocations."""
        traces = super().gather_traces()
        traces["domain"] = self.domain

        # Collect tool invocation histories
        tool_traces = {}
        for name, tool in self.tools.items():
            if hasattr(tool, "_history"):
                tool_traces[name] = {
                    "invocations": tool._history.to_list(),
                }
        traces["tool_invocations"] = tool_traces

        return traces
```

---

## Evaluator Design

### Metrics Mapping

MARBLE Evaluator produces metrics that must be mapped to MASEval format:

| MARBLE Metric | Type | MASEval Mapping |
|---------------|------|-----------------|
| `task_completion` | List[0/1] | `passed` (last value) |
| `token_consumption` | List[int] | `total_tokens` (sum) |
| `planning_score` | List[1-5] | `planning_score` (mean) |
| `communication_score` | List[1-5] | `communication_score` (mean) |
| `code_quality` | Dict | `code_quality` (pass-through) |
| `agent_kpis` | Dict[agent_id, int] | `agent_kpis` (pass-through) |

### MultiAgentBenchEvaluator

```python
class MultiAgentBenchEvaluator(Evaluator):
    """Wraps MARBLE's Evaluator for MASEval integration."""

    def __init__(
        self,
        task: Task,
        environment: MultiAgentBenchEnvironment,
        agents: Sequence[AgentAdapter],
    ):
        self.task = task
        self.environment = environment
        self.agents = agents
        self.metrics_config = task.evaluation_data.get("metrics", {})

        # Create MARBLE evaluator
        from .marble.evaluator.evaluator import Evaluator as MarbleEvaluator
        self._marble_evaluator = MarbleEvaluator(metrics_config=self.metrics_config)

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data needed for MARBLE evaluation."""
        # Get agent traces
        agent_traces = traces.get("agents", {})

        # Get environment state
        env_traces = traces.get("environment", {})

        # Get tool invocations
        tool_traces = env_traces.get("tool_invocations", {})

        return {
            "agent_traces": agent_traces,
            "environment": env_traces,
            "tool_invocations": tool_traces,
            "task_id": self.task.id,
        }

    def __call__(
        self,
        traces: Dict[str, Any],
        final_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run MARBLE evaluation and convert to MASEval format."""
        # Extract MARBLE agents from adapters
        marble_agents = [
            adapter._agent for adapter in self.agents
            if hasattr(adapter, "_agent")
        ]

        # Call MARBLE evaluator update (REQUIRED - not automatic)
        if self.environment._marble_env:
            self._marble_evaluator.update(
                environment=self.environment._marble_env,
                agents=marble_agents,
            )

        # Finalize metrics
        self._marble_evaluator.finalize()

        # Get raw metrics
        raw_metrics = self._marble_evaluator.get_metrics()

        # Convert to MASEval format
        return self._convert_metrics(raw_metrics, final_answer)

    def _convert_metrics(
        self,
        raw_metrics: Dict[str, Any],
        final_answer: Optional[str],
    ) -> Dict[str, Any]:
        """Convert MARBLE metrics to MASEval format."""
        task_completion = raw_metrics.get("task_completion", [])
        planning_scores = raw_metrics.get("planning_score", [])
        comm_scores = raw_metrics.get("communication_score", [])
        token_counts = raw_metrics.get("token_consumption", [])

        # Compute aggregates
        passed = task_completion[-1] == 1 if task_completion else False
        planning_score = sum(planning_scores) / len(planning_scores) if planning_scores else 0.0
        communication_score = sum(comm_scores) / len(comm_scores) if comm_scores else 0.0
        total_tokens = sum(token_counts)

        return {
            "passed": passed,
            "task_completion": task_completion[-1] if task_completion else 0,
            "planning_score": planning_score,
            "communication_score": communication_score,
            "total_tokens": total_tokens,
            "code_quality": raw_metrics.get("code_quality", {}),
            "agent_kpis": raw_metrics.get("agent_kpis", {}),
            "total_milestones": raw_metrics.get("total_milestones", 0),
            "raw_metrics": raw_metrics,  # Include raw for debugging
        }
```

---

## Error Classification

Map MARBLE errors to MASEval's `TaskExecutionStatus`:

```python
from maseval import TaskExecutionStatus, AgentError, EnvironmentError

def classify_marble_error(error: Exception) -> TaskExecutionStatus:
    """Classify MARBLE errors into MASEval status codes."""
    error_str = str(error).lower()

    # Infrastructure failures (not agent's fault)
    if any(x in error_str for x in ["docker", "connection", "timeout", "rate limit"]):
        return TaskExecutionStatus.ENVIRONMENT_ERROR

    # MARBLE internal bugs (not agent's fault)
    if "marble" in error_str or "agentgraph" in error_str:
        return TaskExecutionStatus.ENVIRONMENT_ERROR

    # Missing dependencies
    if "import" in error_str or "module" in error_str:
        return TaskExecutionStatus.SETUP_FAILED

    # LLM/API failures
    if any(x in error_str for x in ["api", "openai", "anthropic", "quota"]):
        return TaskExecutionStatus.ENVIRONMENT_ERROR

    # Agent assertion failures, invalid actions
    if any(x in error_str for x in ["assert", "invalid", "not found"]):
        return TaskExecutionStatus.AGENT_ERROR

    # Default to agent error (conservative - holds agent accountable)
    return TaskExecutionStatus.AGENT_ERROR


def wrap_marble_error(error: Exception) -> Exception:
    """Wrap MARBLE exceptions in MASEval exception types."""
    status = classify_marble_error(error)

    if status == TaskExecutionStatus.ENVIRONMENT_ERROR:
        return EnvironmentError(
            str(error),
            component="MARBLE",
        )
    elif status == TaskExecutionStatus.AGENT_ERROR:
        return AgentError(
            str(error),
            component="MarbleAgent",
        )
    else:
        return error
```

---

## Data Loading

### Task Loading with Validation

```python
from pathlib import Path
from typing import List, Optional, Union
import json

VALID_DOMAINS = frozenset({
    "coding", "database", "minecraft", "research",
    "bargaining", "web", "worldsimulation",
})

def load_tasks(
    domain: str,
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[Task]:
    """Load MultiAgentBench tasks from JSONL.

    Args:
        domain: One of the valid domains (see VALID_DOMAINS)
        data_dir: Base data directory (default: auto-detect)
        limit: Maximum number of tasks

    Returns:
        List of Task objects

    Raises:
        ValueError: If domain is invalid or required fields missing
        FileNotFoundError: If data files not found
    """
    # Validate domain
    if domain.lower() not in VALID_DOMAINS:
        raise ValueError(
            f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}"
        )

    # Find data directory
    data_dir = _resolve_data_dir(data_dir)
    jsonl_path = data_dir / domain / f"{domain}_main.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Task data not found: {jsonl_path}\n"
            f"Ensure MARBLE is cloned to multiagentbench/marble/\n"
            f"See multiagentbench/README.md for setup instructions."
        )

    tasks = []
    with jsonl_path.open() as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break

            entry = json.loads(line)
            task = _parse_task_entry(entry, domain, idx)
            tasks.append(task)

    return tasks


def _resolve_data_dir(data_dir: Optional[Path]) -> Path:
    """Resolve the MARBLE data directory."""
    if data_dir:
        return Path(data_dir)

    # Check standard locations
    candidates = [
        Path(__file__).parent / "marble" / "multiagentbench",
        Path.cwd() / "marble" / "multiagentbench",
    ]

    # Check environment variable
    import os
    env_dir = os.environ.get("MARBLE_DATA_DIR")
    if env_dir:
        candidates.insert(0, Path(env_dir))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "MARBLE data directory not found. Either:\n"
        "1. Clone MARBLE to maseval/benchmark/multiagentbench/marble/\n"
        "2. Set MARBLE_DATA_DIR environment variable\n"
        "See README.md for setup instructions."
    )


def _parse_task_entry(
    entry: Dict[str, Any],
    domain: str,
    idx: int,
) -> Task:
    """Parse a JSONL entry into a MASEval Task.

    Raises:
        ValueError: If required fields are missing (fail loudly, no defaults)
    """
    # Required fields - fail if missing
    REQUIRED_FIELDS = ["scenario", "task_id", "task", "agents", "environment", "relationships"]
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        raise ValueError(
            f"Task entry {idx} missing required fields: {missing}\n"
            f"Entry keys: {list(entry.keys())}"
        )

    # Validate agent specifications
    for i, agent_spec in enumerate(entry["agents"]):
        if "agent_id" not in agent_spec:
            raise ValueError(
                f"Agent {i} in task {entry['task_id']} missing 'agent_id'\n"
                f"Agent spec: {agent_spec}"
            )

    # Extract task content
    task_content = entry["task"]
    if isinstance(task_content, dict):
        query = task_content.get("content", "")
    else:
        query = str(task_content)

    if not query:
        raise ValueError(
            f"Task {entry['task_id']} has empty query/content"
        )

    return Task(
        id=f"{domain}_{entry['task_id']}",
        query=query,
        environment_data={
            "scenario": entry["scenario"],
            "coordinate_mode": entry.get("coordinate_mode", "star"),
            "relationships": entry["relationships"],
            "environment": entry["environment"],
            "task": entry["task"],
            "agents": entry["agents"],
            "max_iterations": entry.get("max_iterations", 10),
            # Store raw entry for MARBLE compatibility
            "raw_marble_config": entry,
        },
        evaluation_data={
            "metrics": entry.get("metrics", {}),
        },
        metadata={
            "domain": domain,
            "task_id": entry["task_id"],
        },
    )


def configure_model_ids(
    tasks: List[Task],
    *,
    agent_model_id: str,
    evaluator_model_id: Optional[str] = None,
) -> List[Task]:
    """Configure model IDs for MARBLE agents and evaluator.

    Args:
        tasks: List of Tasks
        agent_model_id: Model for all MARBLE agents
        evaluator_model_id: Optional model for LLM-based evaluation

    Returns:
        Tasks with model IDs configured (mutated in place)
    """
    for task in tasks:
        # Set agent model
        task.environment_data["llm"] = agent_model_id

        # Set evaluator model
        if evaluator_model_id:
            task.evaluation_data["model_id"] = evaluator_model_id

    return tasks
```

---

## Design Principle Compliance

### R1: Reuse MASEval ✅

- Uses `Benchmark`, `Environment`, `Evaluator`, `AgentAdapter` base classes
- Leverages callback system for tracing
- Uses `Task` data structures
- Benefits from parallel execution, progress bars, error handling

### R2: Scientific Fidelity ✅

- Version pinning via commit hash
- Task data preserved from original JSONL format
- Same coordination strategies available
- MARBLE's agent logic preserved in adapters

### R3: Fail Loudly ✅

- No silent fallbacks that change benchmark results
- Explicit validation with clear error messages
- Infrastructure requirements checked upfront
- Missing fields cause immediate failure

### R4: Maintainability ✅

- Clear module boundaries
- Per-agent adapters match tau2/macs pattern
- Documentation of MARBLE quirks
- Update process documented

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `maseval/benchmark/multiagentbench/` directory structure
- [ ] Add `.gitignore` excluding `marble/`
- [ ] Write `README.md` with MARBLE setup instructions
- [ ] Write `PROVENANCE.md` documenting MARBLE version and license
- [ ] Implement `_resolve_data_dir()` for flexible data location

### Phase 2: Data Loading
- [ ] Implement `load_tasks()` with strict validation
- [ ] Implement `_parse_task_entry()` with required field checks
- [ ] Implement `configure_model_ids()`
- [ ] Unit tests for data loading (valid/invalid inputs)

### Phase 3: Environment
- [ ] Implement `MultiAgentBenchEnvironment`
- [ ] Implement `_create_marble_environment()` for each supported domain
- [ ] Implement `_check_infrastructure()` for Docker-dependent domains
- [ ] Implement `create_tools()` with tracing wrappers
- [ ] Implement `gather_traces()` with tool invocation histories
- [ ] Unit tests for environment setup

### Phase 4: Agent Adapter
- [ ] Implement `MarbleAgentAdapter`
- [ ] Implement `_extract_message_history()` using `msg_box`
- [ ] Implement `gather_traces()` with relationships
- [ ] Implement `_extract_relationships()` from AgentGraph
- [ ] Unit tests for adapter

### Phase 5: Benchmark Class
- [ ] Implement `MultiAgentBenchBenchmark` (abstract base)
  - [ ] `setup_environment()` - create MultiAgentBenchEnvironment
  - [ ] `setup_user()` - return None (no user simulation)
  - [ ] `setup_agents()` - ABSTRACT
  - [ ] `setup_evaluators()` - create MultiAgentBenchEvaluator
- [ ] Implement `MarbleMultiAgentBenchBenchmark`
  - [ ] `setup_agents()` - create agents, AgentGraph, wrap in adapters
  - [ ] `_create_agents()` - instantiate MARBLE BaseAgent instances
  - [ ] `_build_graph_config()` - create AgentGraph config from task
  - [ ] `run_agents()` - implement coordination modes
- [ ] Implement coordination methods:
  - [ ] `_star_coordinate()`
  - [ ] `_graph_coordinate()`
  - [ ] `_tree_coordinate()` (optional - defer if complex)
  - [ ] `_chain_coordinate()` (document MARBLE bug, implement workaround)

### Phase 6: Evaluator
- [ ] Implement `MultiAgentBenchEvaluator`
- [ ] Implement `filter_traces()` - extract agent/env/tool data
- [ ] Implement `__call__()` with explicit `evaluator.update()` call
- [ ] Implement `_convert_metrics()` - MARBLE → MASEval format
- [ ] Unit tests for evaluator

### Phase 7: Error Handling
- [ ] Implement `classify_marble_error()`
- [ ] Implement `wrap_marble_error()`
- [ ] Add try/except wrappers in adapter and benchmark
- [ ] Test error classification

### Phase 8: Integration Testing
- [ ] Create `tests/test_benchmarks/test_multiagentbench/`
- [ ] Integration test: Run 1 Research task
- [ ] Integration test: Run 1 Bargaining task
- [ ] Integration test: Run 1 Coding task
- [ ] Verify traces collected correctly
- [ ] Verify metrics computed correctly
- [ ] Verify error handling works

### Phase 9: Example & Documentation
- [ ] Create `examples/multiagentbench_marble.py`
- [ ] Update main MASEval README with MultiAgentBench section
- [ ] Document known limitations (chain mode bug, Minecraft not supported)

---

## Setup Instructions

### Getting MARBLE Source

```bash
cd maseval/benchmark/multiagentbench
git clone https://github.com/ulab-uiuc/MARBLE.git marble
cd marble
git checkout <pinned-commit-hash>  # Pin to tested version
```

**Recommended:** Pin to a specific commit after testing.

### PROVENANCE.md Template

```markdown
# MARBLE Integration Provenance

- **Source**: https://github.com/ulab-uiuc/MARBLE
- **Version**: Commit `<commit-hash>` (YYYY-MM-DD)
- **License**: MIT (Copyright 2024 Haofei Yu)
- **Vendoring**: Permitted by MIT license with attribution
- **Paper**: arXiv:2503.01935
- **Last Updated**: YYYY-MM-DD

## Known Issues in MARBLE

1. `AgentGraph.get_agent_profiles_linked()` does not exist - breaks chain coordination
2. SharedMemory is per-agent, not actually shared

## Local Patches Applied

- None (document any patches here)
```

---

## Example Usage

```python
from maseval.benchmark.multiagentbench import (
    load_tasks,
    configure_model_ids,
    MarbleMultiAgentBenchBenchmark,
)

# Load tasks from a simple domain (no Docker required)
tasks = load_tasks("research", limit=5)
configure_model_ids(tasks, agent_model_id="gpt-4o")

# Create benchmark
class MyMarbleBenchmark(MarbleMultiAgentBenchBenchmark):
    def get_model_adapter(self, model_id, **kwargs):
        from maseval.interface.openai import OpenAIModelAdapter
        adapter = OpenAIModelAdapter(model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

# Run
benchmark = MyMarbleBenchmark()
results = benchmark.run(tasks)

# Print results
for result in results:
    print(f"Task: {result['task_id']}")
    print(f"Status: {result['status']}")
    if result['eval']:
        print(f"Passed: {result['eval'][0]['passed']}")
        print(f"Planning Score: {result['eval'][0]['planning_score']:.2f}")
```

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| MARBLE API changes | High | High | Pin commit hash, vendored source allows local patches |
| Chain coordination bug | Confirmed | Medium | Document, implement MASEval workaround |
| Docker-dependent domains | Medium | Low | Clear error messages, skip logic, optional support |
| Evaluation discrepancies | Medium | High | Validation study comparing standalone vs MASEval |

---

**Document Version**: 3.0
**Date**: 2026-01-19
**Status**: Ready for Implementation
