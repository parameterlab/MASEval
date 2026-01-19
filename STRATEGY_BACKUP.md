# MARBLE Integration Strategy for MASEval

## Executive Summary

This document proposes integration strategies for bringing MARBLE (Multi-Agent Coordination Backbone with LLM Engine) and its MultiAgentBench benchmark suite into MASEval. After analyzing both codebases, I recommend **Approach 2: Hybrid Vendoring with Adapter Layer** as it best balances scientific fidelity, maintainability, and reusability of MASEval infrastructure.

**Key Findings**:
1. The tau2/macs patterns transfer smoothly to MultiAgentBench with one critical architectural difference: MARBLE's multi-agent engine requires wrapping as an AgentAdapter rather than implementing individual agent setup.
2. The architecture must support **two complementary purposes**:
   - **Exact MARBLE reproduction** via `MarbleMultiAgentBenchBenchmark` (for scientific validation)
   - **Fair framework comparison** via abstract `MultiAgentBenchBenchmark` base (for LangGraph, smolagents, CrewAI implementations)
3. This dual-purpose design enables critical research questions: "Which multi-agent framework performs best on the same tasks?"

---

## Background

### What is MARBLE?

MARBLE is a multi-agent coordination framework that:
- Orchestrates multiple LLM-based agents using an **Engine** + **AgentGraph** architecture
- Supports various coordination modes (chain, hierarchical, graph-based)
- Provides **SharedMemory** for inter-agent communication
- Includes domain-specific environments (Coding, Database, Minecraft, Research)
- Uses YAML-based configuration for agent relationships and task specifications

### What is MultiAgentBench?

MultiAgentBench is MARBLE's benchmark suite featuring:
- **5 domains**: Coding, Database, Minecraft, Research, Bargaining
- **JSONL task format** with rich metadata (agent relationships, coordination modes, evaluation metrics)
- **500+ tasks** testing collaboration and competition scenarios
- Original paper: "MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents" (arXiv:2503.01935)

### MASEval Integration Goals

1. Enable reproduction of MARBLE's published results
2. Allow comparative evaluation of different multi-agent architectures
3. Maintain MASEval's framework-agnostic design
4. Reuse existing MASEval infrastructure (callbacks, tracing, parallelization)

---

## Approach 1: Full Dependency Integration

### Overview
Install MARBLE as a Python package dependency via `pyproject.toml`.

### Architecture
```
maseval/
├── benchmark/
│   └── multiagentbench/
│       ├── __init__.py
│       ├── multiagentbench.py          # Benchmark classes
│       ├── environment.py              # Environment wrapper
│       ├── evaluator.py                # Evaluator wrapper
│       ├── data_loader.py              # Task loading utilities
│       └── adapters/
│           └── marble_adapter.py       # MarbleAgentAdapter
```

**MARBLE installed as dependency:**
```toml
[project.optional-dependencies]
multiagentbench = ["marble-bench>=0.1.0"]
```

### Implementation Pattern

#### 1. Environment Wrapper
```python
class MultiAgentBenchEnvironment(Environment):
    """Wraps MARBLE environment instances."""

    def __init__(self, task_data: Dict[str, Any], callbacks=None):
        self.domain = task_data["scenario"]
        self.env_config = task_data["environment"]
        super().__init__(task_data, callbacks)

    def setup_state(self, task_data: Dict[str, Any]) -> Any:
        # Convert MASEval task data to MARBLE environment config
        return {
            "domain": task_data["scenario"],
            "workspace_dir": task_data["environment"].get("workspace_dir", "workspace"),
            "max_iterations": task_data["environment"].get("max_iterations", 10),
        }

    def create_tools(self) -> Dict[str, Any]:
        # MARBLE environments manage tools internally
        # Return empty dict as tools are accessed through MARBLE Engine
        return {}
```

#### 2. Agent Adapter (Critical Component)
```python
class MarbleAgentAdapter(AgentAdapter):
    """Wraps MARBLE's multi-agent Engine as a single agent."""

    def __init__(
        self,
        engine: "marble.engine.Engine",
        name: str = "marble_engine",
        callbacks: Optional[List[AgentCallback]] = None
    ):
        """
        Args:
            engine: MARBLE Engine instance (pre-configured with agents, graph, memory)
            name: Adapter name for tracing
            callbacks: Optional callbacks
        """
        self.engine = engine
        super().__init__(engine, name, callbacks)

    def _run_agent(self, query: str) -> Any:
        """Execute MARBLE's multi-agent coordination."""
        # MARBLE's Engine.start() runs the full multi-agent workflow
        self.engine.start()

        # Extract final output from SharedMemory or designated output agent
        final_output = self.engine.memory.get("final_answer")

        # Convert MARBLE's execution trace to MessageHistory
        messages = self._convert_to_message_history(self.engine)
        self.messages = messages

        return final_output

    def _convert_to_message_history(self, engine) -> MessageHistory:
        """Convert MARBLE agent traces to MASEval MessageHistory format."""
        messages = []

        # Extract messages from all agents in the engine
        for agent in engine.agents:
            agent_messages = agent.get_messages()  # Assume MARBLE agents track messages
            for msg in agent_messages:
                messages.append({
                    "role": f"agent_{agent.agent_id}",
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "agent_id": agent.agent_id
                })

        return MessageHistory(messages)

    def gather_traces(self) -> Dict[str, Any]:
        """Gather MARBLE-specific traces."""
        return {
            **super().gather_traces(),
            "coordination_mode": self.engine.coordinate_mode,
            "agent_graph": self.engine.graph.to_dict(),
            "shared_memory": self.engine.memory.to_dict(),
            "iterations": self.engine.current_iteration,
            "max_iterations": self.engine.max_iterations,
        }
```

#### 3. Benchmark Harness
```python
class MultiAgentBenchBenchmark(Benchmark):
    """Framework-agnostic MultiAgentBench benchmark.

    Users must implement get_model_adapter() for their LLM provider.
    For MARBLE reproduction, use MarbleMultiAgentBenchBenchmark.
    """

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task
    ) -> MultiAgentBenchEnvironment:
        return MultiAgentBenchEnvironment(
            task_data=task.environment_data,
        )

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: MultiAgentBenchEnvironment,
        task: Task
    ) -> Optional[User]:
        # MultiAgentBench doesn't use external user simulation
        return None

    @abstractmethod
    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: MultiAgentBenchEnvironment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Must be implemented by subclass for specific multi-agent framework."""
        pass

    def setup_evaluators(
        self,
        environment: MultiAgentBenchEnvironment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
    ) -> Sequence[Evaluator]:
        return [
            MultiAgentBenchEvaluator(
                task=task,
                environment=environment,
            )
        ]

    def run_agents(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: MultiAgentBenchEnvironment,
        query: str = "",
    ) -> Any:
        # For MultiAgentBench, run single "agent" (which is the multi-agent engine)
        return agents[0].run(query)


class MarbleMultiAgentBenchBenchmark(MultiAgentBenchBenchmark):
    """MARBLE-specific implementation for reproduction."""

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: MultiAgentBenchEnvironment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create MARBLE Engine and wrap as single agent."""
        from marble.configs.config import Config
        from marble.engine.engine import Engine

        # Convert MASEval task to MARBLE config
        marble_config = self._build_marble_config(agent_data, task)

        # Create MARBLE Engine (contains multiple agents internally)
        engine = Engine(marble_config)

        # Wrap engine as single AgentAdapter
        engine_adapter = MarbleAgentAdapter(
            engine=engine,
            name="marble_engine",
        )

        # Return as single agent (MASEval runs this one adapter)
        return [engine_adapter], {"marble_engine": engine_adapter}

    def _build_marble_config(
        self,
        agent_data: Dict[str, Any],
        task: Task
    ) -> "Config":
        """Convert MASEval Task to MARBLE Config."""
        from marble.configs.config import Config

        config_dict = {
            "coordinate_mode": task.environment_data.get("coordinate_mode", "chain"),
            "relationships": task.environment_data.get("relationships", []),
            "llm": agent_data.get("model_id", "gpt-4"),
            "environment": task.environment_data.get("environment", {}),
            "task": task.environment_data.get("task", {}),
            "agents": task.environment_data.get("agents", []),
            "memory": task.environment_data.get("memory", {"type": "SharedMemory"}),
            "metrics": task.evaluation_data.get("metrics", {}),
            "engine_planner": task.environment_data.get("engine_planner", {}),
        }

        return Config.from_dict(config_dict)
```

#### 4. Data Loading
```python
def load_tasks(
    domain: str,
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> TaskQueue:
    """Load MultiAgentBench tasks from JSONL.

    Args:
        domain: One of "coding", "database", "minecraft", "research", "bargaining"
        data_dir: Base data directory (default: multiagentbench package data)
        limit: Maximum number of tasks to load

    Returns:
        TaskQueue containing Task objects
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    jsonl_path = data_dir / f"{domain}_main.jsonl"

    tasks = []
    with jsonl_path.open() as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break

            entry = json.loads(line)

            # Convert JSONL entry to MASEval Task
            task = Task(
                id=f"{domain}_{entry['task_id']}",
                query=entry["task"]["content"],
                environment_data={
                    "scenario": entry["scenario"],
                    "coordinate_mode": entry["coordinate_mode"],
                    "relationships": entry["relationships"],
                    "environment": entry["environment"],
                    "task": entry["task"],
                    "agents": entry["agents"],
                    "memory": entry["memory"],
                    "engine_planner": entry.get("engine_planner", {}),
                },
                evaluation_data={
                    "metrics": entry.get("metrics", {}),
                },
                metadata={
                    "domain": domain,
                    "task_id": entry["task_id"],
                    "llm": entry.get("llm", ""),
                },
            )
            tasks.append(task)

    return TaskQueue(tasks)


def configure_model_ids(
    tasks: Union[TaskQueue, List[Task]],
    *,
    agent_model_id: str,
    evaluator_model_id: Optional[str] = None,
) -> Union[TaskQueue, List[Task]]:
    """Configure model IDs for MARBLE agents and evaluator."""
    for task in tasks:
        # Set model for MARBLE agents
        task.environment_data["llm"] = agent_model_id

        # Set evaluator model if LLM-based evaluation needed
        if evaluator_model_id:
            task.evaluation_data["evaluate_llm"] = evaluator_model_id

    return tasks
```

### Pros
✅ **Clean separation**: MARBLE remains an independent package
✅ **Standard Python packaging**: Uses established dependency management
✅ **Easy updates**: Can upgrade MARBLE version via `uv add --optional multiagentbench marble-bench@latest`
✅ **R4 (Maintainability)**: Clear boundary between MASEval and MARBLE code

### Cons
❌ **MARBLE not on PyPI**: Would need to install from Git or build local wheel
❌ **Version pinning challenges**: MARBLE development may break compatibility
❌ **Dependency bloat**: MARBLE's dependencies become transitive dependencies
❌ **Limited control**: Can't patch MARBLE bugs without forking

### Design Principle Assessment

- **R1 (Reuse MASEval)**: ✅ Excellent - uses all MASEval patterns
- **R2 (Scientific fidelity)**: ✅ Good - depends on MARBLE version stability
- **R3 (Fail loudly)**: ✅ Good - validation at config conversion boundary
- **R4 (Maintainability)**: ⚠️ Moderate - depends on MARBLE API stability

---

## Approach 2: Hybrid Vendoring with Adapter Layer (RECOMMENDED)

### Overview
Clone MARBLE source into `maseval/benchmark/multiagentbench/marble/` (gitignored), create thin adapter layer in MASEval.

### Architecture
```
maseval/
├── benchmark/
│   └── multiagentbench/
│       ├── __init__.py
│       ├── multiagentbench.py          # Benchmark + Evaluator
│       ├── environment.py              # Environment wrapper
│       ├── data_loader.py              # load_tasks(), configure_model_ids()
│       ├── .gitignore                  # Ignore marble/ directory
│       ├── marble/                     # ← Vendored MARBLE source (gitignored)
│       │   ├── __init__.py
│       │   ├── agent/
│       │   ├── engine/
│       │   ├── environments/
│       │   ├── evaluator/
│       │   ├── graph/
│       │   ├── llms/
│       │   ├── memory/
│       │   └── utils/
│       └── README.md                   # Setup instructions
```

**`.gitignore` in `multiagentbench/`:**
```
marble/
```

**`README.md` in `multiagentbench/`:**
```markdown
# MultiAgentBench Integration

## Setup

1. Clone MARBLE into this directory:
   ```bash
   cd maseval/benchmark/multiagentbench
   git clone https://github.com/ulab-uiuc/MARBLE.git marble
   cd marble
   git checkout <specific-commit-hash>  # Pin to tested version
   ```

2. Install MARBLE dependencies:
   ```bash
   uv pip install -r marble/pyproject.toml
   ```

## Data

MultiAgentBench tasks are located in `marble/multiagentbench/`.
```

### Implementation Pattern

Same as Approach 1, but:
- Import from vendored source: `from .marble.engine.engine import Engine`
- Version control via documented commit hash in README
- Can patch MARBLE locally if needed (document patches in `MARBLE_PATCHES.md`)

### Pros
✅ **Full control**: Can patch bugs, optimize, or modify MARBLE as needed
✅ **Reproducibility**: Pin exact MARBLE version via commit hash
✅ **No external dependency**: Works offline, no PyPI/Git availability issues
✅ **Flexible testing**: Can test against multiple MARBLE versions easily
✅ **R2 (Scientific fidelity)**: Guaranteed exact MARBLE behavior

### Cons
❌ **Setup friction**: Users must manually clone MARBLE (documented in README)
❌ **Disk space**: Each MASEval checkout includes full MARBLE source
❌ **Update overhead**: Must manually update vendored copy
❌ **Licensing**: Must verify MARBLE's MIT license allows vendoring

### Design Principle Assessment

- **R1 (Reuse MASEval)**: ✅ Excellent - identical adapter code to Approach 1
- **R2 (Scientific fidelity)**: ✅ Excellent - exact version pinning
- **R3 (Fail loudly)**: ✅ Good - clear setup instructions, missing vendor errors
- **R4 (Maintainability)**: ✅ Good - documented update process, local patches possible

### Why This is Recommended

1. **Scientific Reproducibility**: Pinning exact MARBLE commit ensures bit-for-bit reproduction of results
2. **Development Velocity**: Can iterate on integration without waiting for MARBLE releases
3. **Risk Mitigation**: MARBLE is early-stage (v0.1.0) and may have breaking changes
4. **Pragmatic**: MASEval is a research tool, not production software - vendoring is acceptable

---

## Approach 3: Multi-Framework Comparison Architecture

### Overview
**This is NOT a replacement for MARBLE - it's the comparison mechanism.** The architecture enables:
1. **Exact MARBLE reproduction** via `MarbleMultiAgentBenchBenchmark`
2. **Alternative framework implementations** via custom subclasses (LangGraph, smolagents, CrewAI)
3. **Fair comparison** - all frameworks run on the same tasks, environment, and evaluation

### Architecture
```python
class MultiAgentBenchBenchmark(Benchmark):
    """Abstract base providing task/eval infrastructure for ANY multi-agent framework.

    Subclasses:
    - MarbleMultiAgentBenchBenchmark: Exact MARBLE reproduction
    - LangGraphMultiAgentBenchBenchmark: LangGraph implementation
    - SmolagentsMultiAgentBenchBenchmark: Smolagents implementation
    - CrewAIMultiAgentBenchBenchmark: CrewAI implementation
    """

    @abstractmethod
    def setup_agents(self, agent_data, environment, task, user):
        """Implement multi-agent coordination using your chosen framework."""
        pass
```

**Example: LangGraph implementation for comparison**
```python
class LangGraphMultiAgentBench(MultiAgentBenchBenchmark):
    """LangGraph implementation of MultiAgentBench tasks.

    Uses same tasks, environment, and evaluation as MARBLE,
    but implements multi-agent coordination with LangGraph.
    """

    def setup_agents(self, agent_data, environment, task, user):
        from langgraph.graph import StateGraph

        # Create graph-based multi-agent system
        graph = StateGraph(...)

        # Add agents from task specification
        for agent_spec in task.environment_data["agents"]:
            agent = self._create_langgraph_agent(agent_spec)
            graph.add_node(agent_spec["agent_id"], agent)

        # Add edges based on relationships
        for src, dst, rel_type in task.environment_data["relationships"]:
            graph.add_edge(src, dst)

        compiled_graph = graph.compile()
        adapter = LangGraphAdapter(compiled_graph, "langgraph_system")
        return [adapter], {"langgraph_system": adapter}
```

### Purpose: Enable Fair Multi-Agent Framework Comparison

This architecture enables researchers to ask:
- **"How does MARBLE compare to LangGraph on the same tasks?"**
- **"Which coordination strategy works best for coding tasks?"**
- **"Does hierarchical vs. graph-based coordination matter?"**

All comparisons use:
- ✅ Same task specifications (from JSONL)
- ✅ Same evaluation metrics (code quality, collaboration)
- ✅ Same environment (Coding, DB, Minecraft, etc.)
- ✅ Only difference: multi-agent coordination implementation

### Pros
✅ **Enables scientific comparison**: Test multiple multi-agent architectures fairly
✅ **Preserves MARBLE reproduction**: MarbleMultiAgentBenchBenchmark still exists
✅ **Framework flexibility**: Users can implement with any multi-agent library
✅ **R1 (Reuse MASEval)**: Perfect - pure MASEval patterns
✅ **Research value**: Answers "which multi-agent approach is better?"

### Cons
⚠️ **Implementation effort**: Each framework requires custom adapter
⚠️ **Results divergence**: Different frameworks produce different results (expected)
⚠️ **Coordination semantics**: Not all frameworks can express all coordination patterns

### Design Principle Assessment

- **R1 (Reuse MASEval)**: ✅ Perfect - clean abstraction
- **R2 (Scientific fidelity)**: ✅ **FOR MARBLE** (via MarbleMultiAgentBenchBenchmark), ✅ **FOR COMPARISONS** (same tasks/eval)
- **R3 (Fail loudly)**: ✅ Clear separation between reproduction and comparison
- **R4 (Maintainability)**: ✅ Excellent - each framework is independent

**Verdict**: This is NOT an alternative to MARBLE integration - it's the **comparison layer** that makes MultiAgentBench valuable for multi-agent systems research. The architecture must support BOTH exact MARBLE reproduction AND alternative framework implementations.

---

## Pattern Transfer Analysis: Do tau2/macs Patterns Apply?

### Summary: YES, with one architectural adaptation

The tau2/macs integration patterns transfer smoothly to MultiAgentBench **with one critical difference**: the multi-agent engine must be wrapped as a single `AgentAdapter` rather than setting up individual agents.

### Pattern Comparison

| Component | MACS/Tau2 Pattern | MultiAgentBench Pattern | Transfer Success |
|-----------|-------------------|-------------------------|------------------|
| **Environment** | Wraps domain-specific environment (tools, database) | Wraps MARBLE domain environment (Coding, DB, Minecraft) | ✅ Direct transfer |
| **Evaluator** | filter_traces() + LLM or deterministic eval | MARBLE evaluator metrics (code quality, collaboration) | ✅ Direct transfer |
| **Data Loading** | load_tasks() from JSON/JSONL | load_tasks() from JSONL | ✅ Direct transfer |
| **Model Config** | configure_model_ids() sets LLM per component | configure_model_ids() sets LLM for all MARBLE agents | ✅ Direct transfer |
| **Benchmark Base** | Abstract base, user implements setup_agents() | Abstract base, user implements setup_agents() | ✅ Direct transfer |
| **Agent Setup** | Create individual AgentAdapters | **DIFFERENT**: Wrap entire MARBLE Engine as single adapter | ⚠️ Requires adaptation |

### Key Architectural Difference: MarbleAgentAdapter

**MACS/Tau2 Pattern:**
```python
def setup_agents(self, agent_data, environment, task, user):
    # Create individual agent adapters
    agent1 = MyAgent("supervisor", tools=environment.tools)
    agent2 = MyAgent("worker", tools=environment.tools)

    adapter1 = AgentAdapter(agent1, "supervisor")
    adapter2 = AgentAdapter(agent2, "worker")

    return [adapter1], {"supervisor": adapter1, "worker": adapter2}
```

**MultiAgentBench Pattern (Approach 1 & 2):**
```python
def setup_agents(self, agent_data, environment, task, user):
    # MARBLE Engine contains multiple agents internally
    from marble.engine.engine import Engine
    from marble.configs.config import Config

    config = self._build_marble_config(task)  # Includes agent specs
    engine = Engine(config)  # Engine manages multiple BaseAgents

    # Wrap entire engine as single adapter
    engine_adapter = MarbleAgentAdapter(engine, "marble_engine")

    return [engine_adapter], {"marble_engine": engine_adapter}
```

**Why this works:**
- MASEval's `Benchmark.run_agents()` calls `agent.run(query)` on agents in the returned list
- `MarbleAgentAdapter._run_agent()` delegates to `engine.start()`, which coordinates all internal agents
- Individual agent messages are aggregated in `MarbleAgentAdapter.gather_traces()`
- From MASEval's perspective, the multi-agent system appears as a single "black box" agent

### Benefits of This Pattern

1. **Preserves MARBLE's coordination logic**: No need to reimplement AgentGraph, SharedMemory, EnginePlanner
2. **Clean abstraction boundary**: MASEval orchestrates benchmark execution, MARBLE handles multi-agent dynamics
3. **Scientific fidelity**: Using MARBLE's Engine exactly as designed ensures reproducible results
4. **Framework comparison**: Easy to swap MARBLE for LangGraph, CrewAI, etc. by implementing different adapters

### Limitations

- **Less granular tracing**: Individual agent messages require extracting from Engine internals
- **Single invocation**: MASEval's `max_invocations` doesn't apply to MARBLE's internal iterations
  - **Solution**: MARBLE's `max_iterations` in config controls internal agent rounds
- **Callback integration**: MARBLE's internal agent callbacks don't automatically trigger MASEval callbacks
  - **Solution**: MarbleAgentAdapter can bridge by polling Engine state in `gather_traces()`

### Verdict: Strong Pattern Transfer

✅ **4/5 components** transfer directly
⚠️ **1/5 components** (agent setup) requires architectural adaptation that is **clean and well-motivated**

---

## Critical Implementation Details

### 1. Message History Conversion

**Challenge**: MARBLE's agents track messages internally; MASEval expects `MessageHistory` from `AgentAdapter.get_messages()`.

**Solution**:
```python
class MarbleAgentAdapter(AgentAdapter):
    def _convert_to_message_history(self, engine) -> MessageHistory:
        """Extract and aggregate messages from all MARBLE agents."""
        messages = []

        for agent in engine.agents:
            # Assume MARBLE agents store messages (may need to add this)
            agent_messages = getattr(agent, "messages", [])

            for msg in agent_messages:
                messages.append({
                    "role": f"agent_{agent.agent_id}",
                    "content": str(msg),
                    "agent_id": agent.agent_id,
                    "timestamp": getattr(msg, "timestamp", None),
                })

        # Include shared memory state
        memory_state = engine.memory.to_dict()
        messages.append({
            "role": "system",
            "content": f"Shared Memory State: {json.dumps(memory_state)}",
        })

        return MessageHistory(messages)
```

**Risk**: MARBLE agents may not expose message history.
**Mitigation**: Add message tracking to MARBLE BaseAgent if needed (document patch).

### 2. Environment Tool Integration

**Challenge**: MARBLE environments manage tools internally; MASEval expects `Environment.create_tools()`.

**Solution**: Return empty dict, document that tools are accessed through MARBLE Engine.
```python
class MultiAgentBenchEnvironment(Environment):
    def create_tools(self) -> Dict[str, Any]:
        # MARBLE environments manage tools internally
        # Tools are available to MARBLE agents through Engine
        return {}

    def gather_traces(self) -> Dict[str, Any]:
        """Override to extract tool traces from MARBLE environment."""
        return {
            **super().gather_traces(),
            "marble_env_type": self.domain,
            "marble_tools": self._extract_marble_tool_traces(),
        }
```

**R3 (Fail loudly)**: If user calls `environment.get_tool()`, raise:
```python
def get_tool(self, name: str) -> Optional[Any]:
    raise NotImplementedError(
        "MultiAgentBenchEnvironment tools are managed by MARBLE Engine. "
        "Access tools through MARBLE agents, not directly from environment."
    )
```

### 3. Evaluator Design

**Challenge**: MultiAgentBench uses MARBLE's Evaluator with domain-specific metrics (code_quality, test_coverage, collaboration_effectiveness).

**Solution**: Wrap MARBLE evaluator in MASEval Evaluator pattern.
```python
class MultiAgentBenchEvaluator(Evaluator):
    """Wraps MARBLE's Evaluator for MASEval integration."""

    def __init__(self, task: Task, environment: MultiAgentBenchEnvironment):
        self.task = task
        self.environment = environment
        self.metrics_config = task.evaluation_data.get("metrics", {})

        # Create MARBLE evaluator
        from marble.evaluator.evaluator import Evaluator as MarbleEvaluator
        self.marble_evaluator = MarbleEvaluator(metrics_config=self.metrics_config)

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract MARBLE-specific execution data."""
        # For coding tasks: extract generated code
        # For DB tasks: extract query sequences
        # For collaboration tasks: extract agent interaction patterns

        marble_engine = traces["agents"]["marble_engine"]
        return {
            "engine_traces": marble_engine,
            "environment_state": traces.get("environment", {}),
        }

    def __call__(
        self,
        traces: Dict[str, Any],
        final_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run MARBLE evaluation metrics."""
        # Extract relevant data
        engine_traces = traces["engine_traces"]

        # Call MARBLE evaluator
        results = self.marble_evaluator.evaluate(
            output=final_answer,
            task_spec=self.task.evaluation_data,
            traces=engine_traces,
        )

        return results  # Should include code_quality, test_coverage, etc.
```

### 4. Data Format Validation

**R3 (Fail loudly)**: Validate JSONL→Task conversion catches missing fields.

```python
def load_tasks(domain: str, ...) -> TaskQueue:
    REQUIRED_FIELDS = [
        "scenario", "task_id", "task", "agents",
        "environment", "relationships"
    ]

    for entry in jsonl_entries:
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            raise ValueError(
                f"Task {entry.get('task_id', 'unknown')} missing required fields: {missing}. "
                f"Ensure MultiAgentBench JSONL format is correct."
            )

        # Validate agent specifications
        for agent_spec in entry["agents"]:
            if "agent_id" not in agent_spec:
                raise ValueError(
                    f"Agent in task {entry['task_id']} missing 'agent_id'. "
                    f"Agent spec: {agent_spec}"
                )
```

### 5. Model Configuration

**Pattern**: Similar to MACS, use `configure_model_ids()` to inject runtime model config.

```python
def configure_model_ids(
    tasks: Union[TaskQueue, List[Task]],
    *,
    agent_model_id: str,
    evaluator_model_id: Optional[str] = None,
) -> Union[TaskQueue, List[Task]]:
    """Configure LLM model IDs for MARBLE agents and evaluator.

    Args:
        tasks: TaskQueue or list of Tasks
        agent_model_id: Model ID for all MARBLE agents (e.g., "gpt-4", "claude-sonnet-4.5")
        evaluator_model_id: Optional model ID for LLM-based evaluation metrics

    Returns:
        Tasks with model IDs configured in environment_data and evaluation_data
    """
    for task in tasks:
        # Set global LLM for all MARBLE agents
        task.environment_data["llm"] = agent_model_id

        # Individual agents can override via agent_config["llm"]
        # This is preserved in task.environment_data["agents"]

        # Set evaluator model if LLM-based metrics enabled
        if evaluator_model_id and task.evaluation_data.get("metrics", {}).get("use_llm_eval"):
            task.evaluation_data["evaluate_llm"] = evaluator_model_id

    return tasks
```

---

## Recommendations

### Primary Recommendation: Approach 2 (Hybrid Vendoring)

**Adopt Approach 2** for initial integration due to:

1. **Scientific Fidelity (R2)**: Guarantees exact MARBLE behavior via version pinning
2. **Development Velocity**: Enables rapid iteration without dependency management overhead
3. **Risk Mitigation**: MARBLE is early-stage; vendoring provides stability
4. **Pragmatic**: Research tool priorities favor reproducibility over packaging aesthetics

### Migration Path: Approach 2 → Approach 1

Once MARBLE stabilizes (v1.0+ release, stable API), migrate to Approach 1:

1. Publish MARBLE to PyPI with semantic versioning
2. Replace vendored source with `[project.optional-dependencies]` entry
3. Keep adapter code identical (no MASEval code changes)
4. Document migration in changelog

**This is a low-risk transition** because adapter code is identical between approaches.

### Approach 3 is Essential, Not Optional

**IMPORTANT**: Approach 3 is NOT a replacement for MARBLE reproduction - it's the **reason MultiAgentBench is valuable for research**.

The complete integration must support:
1. ✅ **Exact MARBLE reproduction** via `MarbleMultiAgentBenchBenchmark` (Approach 1 or 2)
2. ✅ **Fair framework comparison** via abstract `MultiAgentBenchBenchmark` base (Approach 3)

**Research value**: Enables questions like:
- "Does MARBLE's coordination outperform LangGraph?"
- "Which multi-agent framework is best for coding tasks?"
- "How important is the coordination strategy vs. the LLM?"

**Implementation**: Approach 3 is simply an abstract base class - it requires minimal code since the infrastructure (Environment, Evaluator, data loading) is shared with MARBLE reproduction.

---

## Design Principle Compliance

### R1: Reuse MASEval Infrastructure ✅

**Fully compliant.** The integration:
- Uses `Benchmark`, `Environment`, `Evaluator`, `AgentAdapter` base classes
- Leverages callback system for tracing
- Uses `TaskQueue` and `Task` data structures
- Benefits from parallel execution, progress bars, error handling
- No reimplementation of MASEval features

### R2: Scientific Fidelity ✅

**Compliant with caveats.**

**Preserved**:
- Uses MARBLE's Engine, AgentGraph, SharedMemory exactly as designed
- Version pinning via vendored source (Approach 2) or Git dependency (Approach 1)
- Task data preserved from original JSONL format

**Trade-offs**:
- **Different execution environment**: MASEval orchestration vs. standalone MARBLE
  - *Mitigation*: MarbleAgentAdapter delegates directly to Engine.start()
  - *Impact*: Minimal - same agent coordination logic runs
- **Aggregated agent messages**: Individual agent traces must be extracted
  - *Mitigation*: Implement comprehensive trace extraction in MarbleAgentAdapter
  - *Impact*: Low - agent interactions are preserved, just formatted differently

**Validation strategy**:
1. Run subset of tasks with both standalone MARBLE and MASEval integration
2. Compare outputs, metrics, and agent behaviors
3. Document any discrepancies
4. If material differences exist, adjust adapter implementation

### R3: Fail Loudly ✅

**Fully compliant.** The integration includes:

**Validation at boundaries**:
```python
# Data loading
if "agent_id" not in agent_spec:
    raise ValueError("Agent missing 'agent_id'")

# Environment tools
def get_tool(self, name):
    raise NotImplementedError("Tools managed by MARBLE Engine")

# Missing config
if not Path("marble").exists():
    raise FileNotFoundError(
        "MARBLE source not found. See multiagentbench/README.md for setup."
    )
```

**No defensive defaults**:
- Missing JSONL fields → crash with clear error
- Invalid agent relationships → propagate MARBLE error
- Missing MARBLE dependency → crash at import

**Clear error messages**:
- Include context (task ID, agent ID, field name)
- Suggest fixes ("See README.md for setup")
- Link to documentation

### R4: Maintainability ✅

**Compliant.** The integration:

**Clear module boundaries**:
- `data_loader.py`: JSONL → Task conversion (0 dependencies on MARBLE internals)
- `environment.py`: Thin wrapper, delegates to MARBLE
- `multiagentbench.py`: Benchmark + Evaluator, well-documented adapter pattern
- `marble/`: Vendored source, not modified (patches documented separately if needed)

**Documentation**:
- `README.md`: Setup instructions, MARBLE version pinning
- `PROVENANCE.md`: Track MARBLE commit hash, upstream changes
- Docstrings: Explain adapter rationale, MARBLE delegation

**Update process**:
```bash
# Update vendored MARBLE
cd maseval/benchmark/multiagentbench/marble
git pull origin main
git checkout <new-commit-hash>

# Test integration
cd ../../../..
pytest tests/test_benchmarks/test_multiagentbench/ -v

# Document update
echo "<new-commit-hash>" > multiagentbench/MARBLE_VERSION.txt
```

**Long-term maintenance**:
- **Low coupling**: Adapter layer isolates MASEval from MARBLE internals
- **Testable**: Can mock MARBLE Engine for unit tests
- **Upgradable**: Switching to Approach 1 (dependency) requires zero adapter code changes

---

## Implementation Checklist

### Phase 1: Core Integration (Approach 2)
- [ ] Create `maseval/benchmark/multiagentbench/` directory
- [ ] Add `.gitignore` to exclude `marble/`
- [ ] Write `README.md` with MARBLE setup instructions
- [ ] Implement `MultiAgentBenchEnvironment(Environment)`
- [ ] Implement `MarbleAgentAdapter(AgentAdapter)`
  - [ ] `_run_agent()`: Delegate to Engine.start()
  - [ ] `_convert_to_message_history()`: Extract agent messages
  - [ ] `gather_traces()`: Include AgentGraph, SharedMemory
- [ ] Implement `MultiAgentBenchBenchmark(Benchmark)` (abstract base)
- [ ] Implement `MarbleMultiAgentBenchBenchmark(MultiAgentBenchBenchmark)`
  - [ ] `setup_agents()`: Create MARBLE Engine, wrap in adapter
  - [ ] `_build_marble_config()`: Convert Task → MARBLE Config
- [ ] Implement `MultiAgentBenchEvaluator(Evaluator)`
  - [ ] Wrap MARBLE's Evaluator
  - [ ] Extract metrics (code_quality, test_coverage, collaboration)

### Phase 2: Data Loading
- [ ] Implement `load_tasks(domain, limit)` in `data_loader.py`
  - [ ] Validate JSONL fields (R3: Fail loudly)
  - [ ] Convert to Task objects
- [ ] Implement `configure_model_ids(tasks, agent_model_id, evaluator_model_id)`
- [ ] Add tests for data loading with sample JSONL

### Phase 3: Testing & Validation
- [ ] Create `tests/test_benchmarks/test_multiagentbench/`
- [ ] Unit tests for data loading
- [ ] Unit tests for Config conversion
- [ ] Integration test: Run 1 task from each domain
- [ ] Validation test: Compare results with standalone MARBLE (same task, same model)

### Phase 4: Documentation & Examples
- [ ] Example script: `examples/multiagentbench_marble.py`
- [ ] Document setup in main MASEval README
- [ ] Add to documentation site (if exists)
- [ ] Write `PROVENANCE.md`: Track MARBLE version, license, upstream

### Phase 5: Alternative Framework Examples (Essential for Research Value)
- [ ] Implement `LangGraphMultiAgentBench` as example of alternative multi-agent framework
- [ ] Implement `SmolagentsMultiAgentBench` as second comparison point
- [ ] Shows how users can bring their own multi-agent framework
- [ ] Enables comparative research: "MARBLE vs LangGraph vs Smolagents on same tasks"
- [ ] Demonstrates that `MultiAgentBenchBenchmark` abstract base works for ANY framework
- [ ] Document performance comparison methodology

---

## Risks and Mitigations

### Risk 1: MARBLE API Changes
**Probability**: High (early-stage project)
**Impact**: High (breaks integration)
**Mitigation**:
- Pin exact commit hash in README
- Approach 2 (vendoring) allows local patches
- Automated tests detect breakage
- Document MARBLE version compatibility matrix

### Risk 2: Message History Extraction
**Probability**: Medium (depends on MARBLE internals)
**Impact**: Medium (affects tracing quality)
**Mitigation**:
- Test message extraction thoroughly
- If MARBLE agents don't expose messages, contribute PR upstream
- Fallback: Extract from SharedMemory state instead

### Risk 3: License Compatibility
**Probability**: None (VERIFIED ✅)
**Impact**: N/A
**Verification**:
- ✅ MARBLE uses MIT License (Copyright 2024 Haofei Yu)
- ✅ MIT explicitly allows: "use, copy, modify, merge, publish, distribute" without restriction
- ✅ Only requirement: Include copyright notice and license file
**Implementation**:
- Keep `LICENSE` file in vendored `marble/` directory
- Document in `PROVENANCE.md`: "MARBLE is MIT licensed, vendoring permitted with attribution"

### Risk 4: User Setup Friction
**Probability**: High (manual MARBLE clone)
**Impact**: Low (one-time setup)
**Mitigation**:
- Clear README instructions
- Provide setup script: `bash setup_multiagentbench.sh`
- Fail fast with helpful error if MARBLE missing
- Future: Migrate to Approach 1 when MARBLE stabilizes

### Risk 5: Evaluation Discrepancies
**Probability**: Medium (different execution context)
**Impact**: High (invalidates scientific fidelity)
**Mitigation**:
- Run validation study: MASEval vs. standalone MARBLE
- Compare metrics, outputs, agent behaviors
- Document any differences
- Adjust adapter if needed
- If irreconcilable, clearly document limitations

---

## Conclusion

**Adopt Approach 2 (Hybrid Vendoring)** as the recommended integration strategy for MARBLE/MultiAgentBench into MASEval.

This approach:
- ✅ Satisfies all four design principles (R1-R4)
- ✅ Enables reproduction of MARBLE's published results (via `MarbleMultiAgentBenchBenchmark`)
- ✅ Enables fair multi-agent framework comparison (via abstract `MultiAgentBenchBenchmark` base)
- ✅ Provides version stability during MARBLE's early development
- ✅ Reuses MASEval infrastructure effectively
- ✅ Maintains clear architectural boundaries
- ✅ Allows future migration to dependency-based approach

**Key architectural insights**:

1. **MARBLE's multi-agent Engine must be wrapped as a single AgentAdapter**, not decomposed into individual agents. This preserves MARBLE's coordination logic while integrating cleanly with MASEval's orchestration framework.

2. **The architecture must support dual purposes**: Exact MARBLE reproduction (for scientific validation) AND alternative framework implementations (for comparative research). This is what makes MultiAgentBench valuable - researchers can ask "Which multi-agent approach performs better?" on the same task set.

3. **License verified**: MARBLE's MIT license explicitly permits vendoring with attribution (copyright notice + LICENSE file in vendored directory).

The tau2/macs integration patterns transfer successfully with this one architectural adaptation, demonstrating that MASEval's design is flexible enough to accommodate diverse benchmarking paradigms - including multi-agent systems research.

---

## Appendices

### Appendix A: Complete Code Example

See implementation checklist for full module structure. Key classes:
- `MarbleAgentAdapter`: Wraps Engine as single agent
- `MarbleMultiAgentBenchBenchmark`: Creates Engine from Task
- `MultiAgentBenchEvaluator`: Wraps MARBLE metrics
- `load_tasks()`: JSONL → Task conversion

### Appendix B: MARBLE Architecture Summary

- **Engine**: Main orchestrator
- **BaseAgent**: Individual agent with LLM
- **AgentGraph**: Manages agent relationships and coordination
- **SharedMemory**: Inter-agent communication
- **EnginePlanner**: Plans agent execution order
- **Environments**: Domain-specific (Coding, DB, Minecraft, Research, Web, WorldSimulation)
- **Evaluator**: Computes domain-specific metrics

### Appendix C: Task Format Mapping

**JSONL → Task:**
- `task_id` → `Task.id` (prefixed with domain)
- `task.content` → `Task.query`
- `scenario`, `agents`, `environment`, etc. → `Task.environment_data`
- `metrics` → `Task.evaluation_data`
- Remaining fields → `Task.metadata`

### Appendix D: Alternative Multi-Agent Frameworks

After MARBLE integration is stable, users can create custom multi-agent benchmarks:
- LangGraph: Graph-based multi-agent workflows
- CrewAI: Role-based agent teams
- AutoGen: Conversational multi-agent systems
- Custom: Domain-specific coordination logic

Pattern: Subclass `MultiAgentBenchBenchmark`, implement `setup_agents()` with chosen framework.

---

**Document Version**: 1.0
**Date**: 2026-01-19
**Author**: Claude (Sonnet 4.5)
**Status**: Final Recommendation
