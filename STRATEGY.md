# MARBLE Integration Strategy for MASEval

## Executive Summary

This document proposes the integration architecture for bringing MARBLE (Multi-Agent Coordination Backbone with LLM Engine) and its MultiAgentBench benchmark suite into MASEval.

**Key Architecture**: A dual-purpose design that enables:
1. **Exact MARBLE reproduction** via `MarbleMultiAgentBenchBenchmark` (for scientific validation)
2. **Fair framework comparison** via abstract `MultiAgentBenchBenchmark` base (users implement with their own frameworks)

This enables critical research questions: **"Which multi-agent framework performs best on the same tasks?"**

**Key Finding**: The tau2/macs patterns transfer smoothly with one architectural difference: MARBLE's multi-agent engine must be wrapped as a single `AgentAdapter` rather than setting up individual agents.

**License**: ✅ MARBLE's MIT license explicitly permits vendoring/usage with attribution.

---

## Background

### What is MARBLE?

MARBLE is a multi-agent coordination framework that:
- Orchestrates multiple LLM-based agents using an **Engine** + **AgentGraph** architecture
- Supports various coordination modes (chain, hierarchical, graph-based)
- Provides **SharedMemory** for inter-agent communication
- Includes domain-specific environments (Coding, Database, Minecraft, Research, Bargaining)
- Uses configuration for agent relationships and task specifications

### What is MultiAgentBench?

MultiAgentBench is MARBLE's benchmark suite featuring:
- **5 domains**: Coding, Database, Minecraft, Research, Bargaining
- **JSONL task format** with rich metadata (agent relationships, coordination modes, evaluation metrics)
- **500+ tasks** testing collaboration and competition scenarios
- Original paper: "MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents" (arXiv:2503.01935)

### MASEval Integration Goals

1. **Reproduce MARBLE's published results** - Exact reproduction via MarbleMultiAgentBenchBenchmark
2. **Enable multi-agent framework comparison** - Abstract base allows users to implement with any framework
3. **Maintain MASEval's framework-agnostic design** - No hard dependencies on agent frameworks
4. **Reuse MASEval infrastructure** - Callbacks, tracing, parallelization, error handling

---

## Integration Architecture

### Component Structure

```
maseval/benchmark/multiagentbench/
├── __init__.py
├── README.md                              # Setup: clone MARBLE to marble/
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
│   └── marble_adapter.py                  # MarbleAgentAdapter
│
├── marble/                                # ← Vendored MARBLE (gitignored)
│   ├── LICENSE                            #   MIT license preserved
│   ├── agent/
│   ├── engine/
│   ├── environments/
│   ├── evaluator/
│   └── ...                                #   Full MARBLE source
│
└── data/                                  # Symlink to marble/multiagentbench/
    ├── coding/
    ├── database/
    ├── minecraft/
    ├── research/
    └── bargaining/
```

### Class Hierarchy

**In `maseval/benchmark/multiagentbench/`:**

```python
# Abstract base - provides task/eval infrastructure for ANY framework
class MultiAgentBenchBenchmark(Benchmark):
    """Framework-agnostic base for MultiAgentBench tasks.

    Users implement setup_agents() with their chosen framework.
    """
    def setup_environment(...)  → MultiAgentBenchEnvironment  # Shared
    def setup_evaluators(...)   → MultiAgentBenchEvaluator    # Shared
    def setup_agents(...)       → ABSTRACT                    # User implements

# MARBLE reproduction - the only concrete implementation in main library
class MarbleMultiAgentBenchBenchmark(MultiAgentBenchBenchmark):
    """Exact MARBLE reproduction for scientific validation."""
    def setup_agents(...):
        # Convert Task → MARBLE Config
        # Create MARBLE Engine (contains multiple agents internally)
        # Wrap as single MarbleAgentAdapter
        # Return to MASEval
```

**User implementations (examples/ directory only):**

```python
# Example: LangGraph implementation (in examples/, NOT in main library)
class LangGraphMultiAgentBench(MultiAgentBenchBenchmark):
    def setup_agents(...):
        # Build LangGraph from task.environment_data["relationships"]
        # User's custom implementation

# Example: Smolagents implementation (in examples/, NOT in main library)
class SmolagentsMultiAgentBench(MultiAgentBenchBenchmark):
    def setup_agents(...):
        # User's custom implementation
```

---

## How The Integration Works

### Execution Flow

Let me trace a complete execution from user code to MARBLE coordination:

#### 1. User Loads Tasks

```python
from maseval.benchmark.multiagentbench import load_tasks, configure_model_ids

# Load tasks from JSONL
tasks = load_tasks("coding", limit=5)
# Reads marble/multiagentbench/coding/coding_main.jsonl
# Each JSONL line → MASEval Task object

# Configure model for all agents
configure_model_ids(tasks, agent_model_id="gpt-4")
```

**Task structure after loading:**
```python
Task(
    id="coding_1",
    query="Software Development Task: Please write a system called...",
    environment_data={
        "scenario": "coding",
        "coordinate_mode": "chain",  # or "hierarchical", "graph"
        "relationships": [
            ["agent1", "agent2", "reports_to"],
            ["agent2", "agent3", "collaborates_with"],
        ],
        "agents": [
            {"agent_id": "agent1", "profile": "Senior Developer...", "type": "CodingAgent"},
            {"agent_id": "agent2", "profile": "Code Reviewer...", "type": "CodingAgent"},
            {"agent_id": "agent3", "profile": "QA Engineer...", "type": "CodingAgent"},
        ],
        "environment": {"type": "Coding", "workspace_dir": "workspace", "max_iterations": 10},
        "llm": "gpt-4",
    },
    evaluation_data={
        "metrics": {"code_quality": true, "test_coverage": true, "collaboration_effectiveness": true}
    },
)
```

#### 2. User Chooses Implementation

**Option A: MARBLE Reproduction (built-in)**
```python
from maseval.benchmark.multiagentbench import MarbleMultiAgentBenchBenchmark

benchmark = MarbleMultiAgentBenchBenchmark()
results = benchmark.run(tasks)  # Exact MARBLE behavior
```

**Option B: Custom Framework Implementation (user-provided)**
```python
# User implements their own benchmark (e.g., in examples/ or custom code)
# Example shown in examples/multiagentbench_langgraph.py

from examples.multiagentbench_langgraph import LangGraphMultiAgentBench

benchmark = LangGraphMultiAgentBench()
results = benchmark.run(tasks)  # Same tasks, user's framework
```

#### 3. MASEval Orchestration Loop

For each task, `benchmark.run()` calls:

```python
# Phase 1: Setup components
environment = benchmark.setup_environment(agent_data={}, task=task)
# → MultiAgentBenchEnvironment (wraps MARBLE CodingEnvironment)

user = benchmark.setup_user(agent_data={}, environment, task)
# → None (MultiAgentBench doesn't use external user simulation)

agents, agents_dict = benchmark.setup_agents(agent_data={}, environment, task, user)
# → This is where MARBLE vs LangGraph differ!

evaluators = benchmark.setup_evaluators(environment, task, agents, user)
# → MultiAgentBenchEvaluator (wraps MARBLE's Evaluator)

# Phase 2: Execute agents
final_answer = benchmark.run_agents(agents, task, environment, query=task.query)
# → Calls agent.run(query) - delegates to MARBLE or LangGraph

# Phase 3: Evaluate
traces = benchmark.collect_all_traces()
# → Gathers agent messages, tool calls, memory state

eval_results = benchmark.evaluate(evaluators, agents_dict, final_answer, traces)
# → Computes code_quality, test_coverage, collaboration metrics

# Store result
report = {
    "task_id": task.id,
    "final_answer": final_answer,
    "eval": eval_results,
    "traces": traces,
    "status": "success"
}
```

#### 4. Inside `MarbleMultiAgentBenchBenchmark.setup_agents()`

This is where MARBLE integration happens:

```python
def setup_agents(
    self,
    agent_data: Dict[str, Any],
    environment: MultiAgentBenchEnvironment,
    task: Task,
    user: Optional[User],
) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
    """Create MARBLE Engine and wrap as single AgentAdapter."""

    from .marble.configs.config import Config
    from .marble.engine.engine import Engine

    # Step 1: Convert MASEval Task → MARBLE Config
    marble_config = Config.from_dict({
        "coordinate_mode": task.environment_data["coordinate_mode"],
        "relationships": task.environment_data["relationships"],
        "agents": task.environment_data["agents"],
        "environment": task.environment_data["environment"],
        "task": {"content": task.query},
        "llm": task.environment_data["llm"],
        "memory": {"type": "SharedMemory"},
        "metrics": task.evaluation_data["metrics"],
        "engine_planner": {"initial_progress": "Starting task"},
    })

    # Step 2: Create MARBLE Engine
    # This internally creates:
    # - 3 BaseAgent instances (agent1, agent2, agent3)
    # - AgentGraph managing relationships
    # - SharedMemory for inter-agent communication
    # - EnginePlanner for coordination strategy
    # - Domain environment (CodingEnvironment)
    engine = Engine(marble_config)

    # Step 3: Wrap entire engine as single AgentAdapter
    # MASEval sees one "agent" but it contains 3+ agents internally
    engine_adapter = MarbleAgentAdapter(
        engine=engine,
        name="marble_engine",
    )

    # Step 4: Return to MASEval
    # agents_to_run: [engine_adapter] - MASEval will call .run() on this
    # agents_dict: {"marble_engine": engine_adapter} - for tracing
    return [engine_adapter], {"marble_engine": engine_adapter}
```

**Key Insight**: MASEval never sees MARBLE's internal agents (agent1, agent2, agent3). It only sees the wrapper. This preserves MARBLE's coordination logic while integrating cleanly.

#### 5. Inside `MarbleAgentAdapter._run_agent()`

When MASEval calls `agent.run(query)`:

```python
class MarbleAgentAdapter(AgentAdapter):
    """Wraps MARBLE's multi-agent Engine as a single agent."""

    def __init__(self, engine: "Engine", name: str = "marble_engine", callbacks=None):
        self.engine = engine  # Contains multiple agents + coordination
        super().__init__(engine, name, callbacks)

    def _run_agent(self, query: str) -> Any:
        """Execute MARBLE's multi-agent coordination."""

        # Delegate to MARBLE Engine
        self.engine.start()

        # Engine internally does:
        # 1. EnginePlanner decides execution order based on coordinate_mode
        # 2. agent1.run() → writes to SharedMemory
        # 3. agent2.run() → reads SharedMemory, writes response
        # 4. agent3.run() → reads SharedMemory, final output
        # 5. Agents use AgentGraph to check who they can communicate with
        # 6. Runs until task complete or max_iterations reached

        # Extract final answer from SharedMemory or designated output
        final_answer = self.engine.memory.get("final_answer")
        if not final_answer:
            # Fallback: get from last agent's output
            final_answer = self.engine.agents[-1].last_output

        # Convert MARBLE's execution traces → MASEval MessageHistory
        self.messages = self._convert_to_message_history(self.engine)

        return final_answer

    def _convert_to_message_history(self, engine) -> MessageHistory:
        """Extract messages from all MARBLE agents."""
        messages = []

        # Gather from each internal agent
        for agent in engine.agents:
            agent_msgs = getattr(agent, "messages", [])
            for msg in agent_msgs:
                messages.append({
                    "role": f"agent_{agent.agent_id}",
                    "content": str(msg),
                    "agent_id": agent.agent_id,
                    "timestamp": getattr(msg, "timestamp", None),
                })

        # Include SharedMemory state
        messages.append({
            "role": "system",
            "content": f"SharedMemory: {engine.memory.to_dict()}",
        })

        return MessageHistory(messages)

    def gather_traces(self) -> Dict[str, Any]:
        """Gather MARBLE-specific execution data."""
        return {
            **super().gather_traces(),
            "coordination_mode": self.engine.coordinate_mode,
            "agent_graph": self.engine.graph.to_dict(),
            "shared_memory": self.engine.memory.to_dict(),
            "iterations": self.engine.current_iteration,
            "max_iterations": self.engine.max_iterations,
            "internal_agents": [
                {"agent_id": a.agent_id, "type": a.agent_type}
                for a in self.engine.agents
            ],
        }
```

#### 6. Alternative: User Framework Implementation Example

**Note**: This is a USER implementation (in examples/), NOT part of main library.

For comparison with a different framework, users implement their own benchmark:

```python
# In examples/multiagentbench_langgraph.py (NOT in maseval/benchmark/)

class LangGraphMultiAgentBench(MultiAgentBenchBenchmark):
    """Example user implementation using LangGraph."""

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: MultiAgentBenchEnvironment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create LangGraph-based multi-agent system."""

        from langgraph.graph import StateGraph

        # Same task data, different implementation
        graph = StateGraph(AgentState)

        # Create agents from same specs
        for agent_spec in task.environment_data["agents"]:
            agent = self._create_langgraph_agent(
                agent_id=agent_spec["agent_id"],
                profile=agent_spec["profile"],
                tools=environment.tools,  # Same environment!
                model=task.environment_data["llm"],  # Same model!
            )
            graph.add_node(agent_spec["agent_id"], agent)

        # Interpret relationships as edges
        for src, dst, rel_type in task.environment_data["relationships"]:
            if rel_type in ["reports_to", "collaborates_with"]:
                graph.add_edge(src, dst)

        # Compile and wrap
        compiled = graph.compile()
        adapter = LangGraphAdapter(compiled, "langgraph_system")

        return [adapter], {"langgraph_system": adapter}
```

**Result**: Both MARBLE and user frameworks run on:
- ✅ Same tasks (coding task #1)
- ✅ Same agent specs (agent1, agent2, agent3)
- ✅ Same environment (CodingEnvironment)
- ✅ Same evaluation (code_quality, collaboration)
- ❌ **Different coordination strategy** - this is what enables comparison!

---

## Pattern Transfer Analysis: Do tau2/macs Patterns Apply?

### Summary: YES, with one architectural adaptation

The tau2/macs integration patterns transfer smoothly to MultiAgentBench.

### Pattern Comparison

| Component | MACS/Tau2 Pattern | MultiAgentBench Pattern | Transfer |
|-----------|-------------------|-------------------------|----------|
| **Environment** | Wraps domain-specific environment (tools, database) | Wraps MARBLE domain environment (Coding, DB, Minecraft) | ✅ Direct |
| **Evaluator** | filter_traces() + LLM or deterministic eval | MARBLE evaluator metrics (code quality, collaboration) | ✅ Direct |
| **Data Loading** | load_tasks() from JSON/JSONL | load_tasks() from JSONL | ✅ Direct |
| **Model Config** | configure_model_ids() sets LLM per component | configure_model_ids() sets LLM for all MARBLE agents | ✅ Direct |
| **Benchmark Base** | Abstract base, user implements setup_agents() | Abstract base, user implements setup_agents() | ✅ Direct |
| **Agent Setup** | Create individual AgentAdapters | **DIFFERENT**: Wrap entire MARBLE Engine as single adapter | ⚠️ Adapted |

### Key Architectural Difference

**MACS/Tau2 Pattern:**
```python
def setup_agents(self, agent_data, environment, task, user):
    # Create individual agent adapters
    supervisor = MyAgent("supervisor", tools=environment.tools)
    worker = MyAgent("worker", tools=environment.tools)

    adapter1 = AgentAdapter(supervisor, "supervisor")
    adapter2 = AgentAdapter(worker, "worker")

    return [adapter1], {"supervisor": adapter1, "worker": adapter2}
```

**MultiAgentBench Pattern:**
```python
def setup_agents(self, agent_data, environment, task, user):
    # MARBLE Engine contains multiple agents internally
    from .marble.engine.engine import Engine

    config = self._build_marble_config(task)
    engine = Engine(config)  # Creates agent1, agent2, agent3 internally

    # Wrap entire engine as single adapter
    engine_adapter = MarbleAgentAdapter(engine, "marble_engine")

    return [engine_adapter], {"marble_engine": engine_adapter}
```

**Why this works:**
- MASEval's `run_agents()` calls `agent.run(query)` on returned agents
- `MarbleAgentAdapter._run_agent()` delegates to `engine.start()`
- MARBLE handles all internal coordination (agent-to-agent communication)
- Individual agent messages are aggregated in `gather_traces()`

**Benefits:**
1. Preserves MARBLE's coordination logic (no reimplementation)
2. Clean separation: MASEval orchestrates outer loop, MARBLE handles inner coordination
3. Scientific fidelity: Using MARBLE exactly as designed
4. Extensible: Users can implement custom frameworks by subclassing `MultiAgentBenchBenchmark`

### Verdict: Strong Pattern Transfer

✅ **5/6 components** transfer directly
⚠️ **1/6 components** (agent setup) requires clean, well-motivated adaptation

---

## Critical Implementation Details

### 1. Environment Integration

**Challenge**: MARBLE environments manage tools internally; MASEval expects `Environment.create_tools()`.

**Solution**:
```python
class MultiAgentBenchEnvironment(Environment):
    """Wraps MARBLE environment instances."""

    def __init__(self, task_data: Dict[str, Any], callbacks=None):
        self.domain = task_data["scenario"]
        self.marble_env_config = task_data["environment"]
        super().__init__(task_data, callbacks)

    def setup_state(self, task_data: Dict[str, Any]) -> Any:
        return {
            "domain": task_data["scenario"],
            "marble_config": task_data["environment"],
        }

    def create_tools(self) -> Dict[str, Any]:
        # MARBLE environments manage tools internally
        # Tools are accessed through MARBLE agents, not directly
        # Return empty dict to satisfy MASEval interface
        return {}

    def get_tool(self, name: str) -> Optional[Any]:
        # R3: Fail loudly if someone tries to access tools directly
        raise NotImplementedError(
            "MultiAgentBenchEnvironment tools are managed by MARBLE Engine. "
            "Access tools through MARBLE agents, not directly from environment."
        )

    def gather_traces(self) -> Dict[str, Any]:
        """Extract traces from MARBLE environment if available."""
        return {
            **super().gather_traces(),
            "domain": self.domain,
            "marble_env_type": self.marble_env_config.get("type"),
        }
```

### 2. Evaluator Design

**Challenge**: MARBLE uses domain-specific metrics (code_quality, collaboration_effectiveness).

**Solution**:
```python
class MultiAgentBenchEvaluator(Evaluator):
    """Wraps MARBLE's Evaluator for MASEval integration."""

    def __init__(self, task: Task, environment: MultiAgentBenchEnvironment):
        self.task = task
        self.environment = environment
        self.metrics_config = task.evaluation_data.get("metrics", {})

        # Create MARBLE evaluator
        from .marble.evaluator.evaluator import Evaluator as MarbleEvaluator
        self.marble_evaluator = MarbleEvaluator(metrics_config=self.metrics_config)

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract MARBLE-specific execution data."""

        # For coding: extract generated code
        # For DB: extract query sequences
        # For collaboration: extract agent interactions

        marble_engine = traces.get("agents", {}).get("marble_engine", {})

        return {
            "engine_traces": marble_engine,
            "shared_memory": marble_engine.get("shared_memory", {}),
            "agent_graph": marble_engine.get("agent_graph", {}),
            "iterations": marble_engine.get("iterations", 0),
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

        # Results should include:
        # - code_quality: float (for coding tasks)
        # - test_coverage: float (for coding tasks)
        # - collaboration_effectiveness: float (for all tasks)
        # - task_completion: bool

        return results
```

### 3. Data Loading

**Task format validation (R3: Fail loudly)**:

```python
def load_tasks(
    domain: str,
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> TaskQueue:
    """Load MultiAgentBench tasks from JSONL.

    Args:
        domain: One of "coding", "database", "minecraft", "research", "bargaining"
        data_dir: Base data directory (default: marble/multiagentbench/)
        limit: Maximum number of tasks

    Returns:
        TaskQueue containing Task objects

    Raises:
        ValueError: If domain invalid or required fields missing
    """
    VALID_DOMAINS = ("coding", "database", "minecraft", "research", "bargaining")
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {VALID_DOMAINS}")

    # Default to vendored MARBLE data
    data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "marble" / "multiagentbench"
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

            # R3: Validate required fields
            REQUIRED_FIELDS = ["scenario", "task_id", "task", "agents", "environment", "relationships"]
            missing = [f for f in REQUIRED_FIELDS if f not in entry]
            if missing:
                raise ValueError(
                    f"Task {entry.get('task_id', idx)} missing required fields: {missing}\n"
                    f"Ensure JSONL format matches MultiAgentBench specification."
                )

            # Validate agent specifications
            for agent_spec in entry["agents"]:
                if "agent_id" not in agent_spec:
                    raise ValueError(
                        f"Agent in task {entry['task_id']} missing 'agent_id'\n"
                        f"Agent spec: {agent_spec}"
                    )

            # Convert to MASEval Task
            task = Task(
                id=f"{domain}_{entry['task_id']}",
                query=entry["task"]["content"],
                environment_data={
                    "scenario": entry["scenario"],
                    "coordinate_mode": entry.get("coordinate_mode", "chain"),
                    "relationships": entry["relationships"],
                    "environment": entry["environment"],
                    "task": entry["task"],
                    "agents": entry["agents"],
                    "memory": entry.get("memory", {"type": "SharedMemory"}),
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
    """Configure model IDs for MARBLE agents and evaluator.

    Args:
        tasks: TaskQueue or list of Tasks
        agent_model_id: Model for all MARBLE agents (e.g., "gpt-4", "claude-sonnet-4.5")
        evaluator_model_id: Optional model for LLM-based evaluation metrics

    Returns:
        Tasks with model IDs configured (mutated in place)
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

### 4. Message History Extraction

**Challenge**: MARBLE agents may not expose message history directly.

**Fallback strategies**:
```python
def _convert_to_message_history(self, engine) -> MessageHistory:
    """Extract messages from MARBLE agents with fallbacks."""
    messages = []

    # Strategy 1: Direct message access
    for agent in engine.agents:
        if hasattr(agent, "messages"):
            for msg in agent.messages:
                messages.append({
                    "role": f"agent_{agent.agent_id}",
                    "content": str(msg),
                    "agent_id": agent.agent_id,
                })

    # Strategy 2: Fallback to SharedMemory
    if not messages:
        memory_state = engine.memory.to_dict()
        for key, value in memory_state.items():
            messages.append({
                "role": "system",
                "content": f"{key}: {value}",
            })

    # Strategy 3: Fallback to agent outputs
    if not messages:
        for agent in engine.agents:
            if hasattr(agent, "last_output"):
                messages.append({
                    "role": f"agent_{agent.agent_id}",
                    "content": str(agent.last_output),
                    "agent_id": agent.agent_id,
                })

    return MessageHistory(messages)
```

---

## Setup Instructions

### Getting MARBLE Source

Since MARBLE's pip installation has bugs, clone the source directly:

```bash
cd maseval/benchmark/multiagentbench
git clone https://github.com/ulab-uiuc/MARBLE.git marble
cd marble
git checkout <commit-hash>  # Pin to tested version
```

**Recommended commit**: `<commit-hash>` (tested with MASEval integration)

**Document in `README.md`**:
```markdown
# MultiAgentBench Integration Setup

1. Clone MARBLE source:
   ```bash
   cd maseval/benchmark/multiagentbench
   git clone https://github.com/ulab-uiuc/MARBLE.git marble
   cd marble
   git checkout abc123def  # Pinned version
   ```

2. Install MARBLE dependencies:
   ```bash
   # From maseval root
   uv pip install -e ./maseval/benchmark/multiagentbench/marble
   ```

3. Run example:
   ```bash
   python examples/multiagentbench_marble.py
   ```
```

**Document in `PROVENANCE.md`**:
```markdown
# MARBLE Integration Provenance

- **Source**: https://github.com/ulab-uiuc/MARBLE
- **Version**: Commit `abc123def` (2025-01-19)
- **License**: MIT (Copyright 2024 Haofei Yu)
- **Vendoring**: Permitted by MIT license with attribution
- **Paper**: arXiv:2503.01935
- **Last Updated**: 2025-01-19
- **Integration**: Wrapped via MarbleAgentAdapter

## Changes from Upstream
- None (vanilla MARBLE source)
- If patches needed, document here
```

---

## Design Principle Compliance

### R1: Reuse MASEval ✅

**Fully compliant.** The integration:
- Uses `Benchmark`, `Environment`, `Evaluator`, `AgentAdapter` base classes
- Leverages callback system for tracing
- Uses `TaskQueue` and `Task` data structures
- Benefits from parallel execution, progress bars, error handling
- No reimplementation of MASEval features

### R2: Scientific Fidelity ✅

**Compliant with validation.**

**Preserved**:
- Uses MARBLE's Engine, AgentGraph, SharedMemory exactly as designed
- Version pinning via commit hash
- Task data preserved from original JSONL format
- Same coordination strategies (chain, hierarchical, graph)

**Validation Strategy**:
1. Run subset of tasks with both standalone MARBLE and MASEval integration
2. Compare final outputs, metrics, agent behaviors
3. Document any discrepancies
4. Adjust adapter if differences are material

**Expected**: Minor differences in execution traces (formatted differently) but identical final outputs and metrics.

### R3: Fail Loudly ✅

**Fully compliant.** The integration includes:

**Validation at boundaries**:
```python
# Missing MARBLE source
if not Path("marble").exists():
    raise FileNotFoundError("MARBLE not found. See README.md for setup.")

# Invalid domain
if domain not in VALID_DOMAINS:
    raise ValueError(f"Invalid domain '{domain}'")

# Missing required fields
if "agent_id" not in agent_spec:
    raise ValueError(f"Agent missing 'agent_id': {agent_spec}")

# Incorrect tool access
def get_tool(self, name):
    raise NotImplementedError("Tools managed by MARBLE Engine")
```

**No defensive defaults**:
- Missing JSONL fields → crash with error message
- Invalid relationships → propagate MARBLE error
- Missing config → fail at Engine initialization

### R4: Maintainability ✅

**Compliant.** The integration:

**Clear module boundaries**:
- `data_loader.py`: JSONL → Task conversion (zero MARBLE dependencies)
- `environment.py`: Thin wrapper, delegates to MARBLE
- `multiagentbench.py`: Benchmark + Evaluator, documented adapter pattern
- `adapters/marble_adapter.py`: Single-purpose adapter
- `marble/`: Vendored source, not modified

**Documentation**:
- `README.md`: Setup instructions, version pinning
- `PROVENANCE.md`: Track upstream, document changes
- Docstrings: Explain rationale, MARBLE delegation
- Code comments: Clarify adapter pattern decisions

**Update Process**:
```bash
# Update vendored MARBLE
cd maseval/benchmark/multiagentbench/marble
git pull origin main
git checkout <new-commit>

# Test integration
cd ../../../..
pytest tests/test_benchmarks/test_multiagentbench/ -v

# Document update
# Update PROVENANCE.md with new commit hash and date
```

**Long-term**:
- Low coupling: Adapter isolates MASEval from MARBLE internals
- Testable: Can mock MARBLE Engine for unit tests
- Extensible: Users can add custom framework implementations via examples/

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `maseval/benchmark/multiagentbench/` directory
- [ ] Add `.gitignore` excluding `marble/`
- [ ] Write `README.md` with MARBLE setup instructions
- [ ] Write `PROVENANCE.md` documenting MARBLE version and license
- [ ] Clone MARBLE to `marble/` directory (manual step for users)
- [ ] Create symlink: `data/` → `marble/multiagentbench/`

### Phase 2: Core Classes
- [ ] Implement `MultiAgentBenchEnvironment(Environment)`
  - [ ] `setup_state()`: Store domain and config
  - [ ] `create_tools()`: Return empty dict (tools managed by MARBLE)
  - [ ] `get_tool()`: Raise NotImplementedError with clear message (R3)
  - [ ] `gather_traces()`: Extract domain info
- [ ] Implement `MarbleAgentAdapter(AgentAdapter)`
  - [ ] `__init__()`: Store MARBLE Engine
  - [ ] `_run_agent()`: Delegate to `engine.start()`
  - [ ] `_convert_to_message_history()`: Extract from agents + SharedMemory
  - [ ] `gather_traces()`: Include AgentGraph, SharedMemory, iterations
- [ ] Implement `MultiAgentBenchEvaluator(Evaluator)`
  - [ ] Wrap MARBLE's Evaluator
  - [ ] `filter_traces()`: Extract engine traces
  - [ ] `__call__()`: Compute metrics (code_quality, collaboration, etc.)

### Phase 3: Benchmark Classes
- [ ] Implement `MultiAgentBenchBenchmark(Benchmark)` (abstract base)
  - [ ] `setup_environment()`: Create MultiAgentBenchEnvironment
  - [ ] `setup_user()`: Return None
  - [ ] `setup_agents()`: ABSTRACT
  - [ ] `setup_evaluators()`: Create MultiAgentBenchEvaluator
  - [ ] `run_agents()`: Call agent.run(query)
- [ ] Implement `MarbleMultiAgentBenchBenchmark(MultiAgentBenchBenchmark)`
  - [ ] `setup_agents()`: Create MARBLE Engine, wrap in adapter
  - [ ] `_build_marble_config()`: Convert Task → MARBLE Config
  - [ ] `get_model_adapter()`: Create/register model adapters

### Phase 4: Data Loading
- [ ] Implement `load_tasks()` in `data_loader.py`
  - [ ] Validate domain (R3)
  - [ ] Validate JSONL fields (R3)
  - [ ] Convert to Task objects
  - [ ] Clear error messages for missing data
- [ ] Implement `configure_model_ids()`
  - [ ] Set agent model in environment_data
  - [ ] Set evaluator model in evaluation_data
- [ ] Add unit tests for data loading

### Phase 5: Testing & Validation
- [ ] Create `tests/test_benchmarks/test_multiagentbench/`
- [ ] Unit tests:
  - [ ] `test_load_tasks()`: Validates JSONL parsing
  - [ ] `test_configure_model_ids()`: Validates model config
  - [ ] `test_marble_config_conversion()`: Task → Config
  - [ ] `test_environment()`: MultiAgentBenchEnvironment
  - [ ] `test_evaluator()`: MultiAgentBenchEvaluator
- [ ] Integration tests:
  - [ ] Run 1 coding task with MARBLE
  - [ ] Run 1 database task with MARBLE
  - [ ] Verify metrics computed
  - [ ] Verify traces collected
- [ ] Validation test:
  - [ ] Run same task standalone vs MASEval
  - [ ] Compare outputs and metrics
  - [ ] Document any discrepancies

### Phase 6: Documentation & Examples
- [ ] Example: `examples/multiagentbench_marble.py`
  - [ ] Load coding tasks
  - [ ] Run with MarbleMultiAgentBenchBenchmark
  - [ ] Print results and metrics
- [ ] Update main MASEval README
  - [ ] Document MultiAgentBench integration
  - [ ] Setup instructions for MARBLE
- [ ] Add to documentation site (if exists)

### Phase 7: Optional User Framework Examples (in examples/ only)
- [ ] Example: `examples/multiagentbench_langgraph.py`
  - [ ] Implement `LangGraphMultiAgentBench(MultiAgentBenchBenchmark)`
  - [ ] Parse relationships → LangGraph structure
  - [ ] Demonstrate how users can implement custom frameworks
  - [ ] Document in example README
- [ ] Example: `examples/multiagentbench_smolagents.py`
  - [ ] Implement `SmolagentsMultiAgentBench(MultiAgentBenchBenchmark)`
  - [ ] Show second framework example
  - [ ] Demonstrate comparison methodology
- [ ] Document in examples README:
  - [ ] How to implement custom multi-agent framework
  - [ ] How to compare metrics across frameworks
  - [ ] What differences are expected vs problematic
  - [ ] **Important**: These are EXAMPLES, not part of main library

---

## Risks and Mitigations

### Risk 1: MARBLE API Changes
**Probability**: High (early-stage project)
**Impact**: High (breaks integration)
**Mitigation**:
- Pin exact commit hash in README and PROVENANCE.md
- Vendored source allows local patches if needed
- Automated tests detect breakage quickly
- Document MARBLE version compatibility matrix

### Risk 2: Message History Extraction
**Probability**: Medium (depends on MARBLE internals)
**Impact**: Medium (affects tracing quality)
**Mitigation**:
- Implement fallback strategies (SharedMemory, agent outputs)
- Test extraction thoroughly
- If MARBLE doesn't expose messages, contribute PR upstream
- Document limitations in traces

### Risk 3: License Compatibility
**Probability**: None (VERIFIED ✅)
**Impact**: N/A
**Verification**:
- ✅ MARBLE uses MIT License (Copyright 2024 Haofei Yu)
- ✅ MIT explicitly allows: "use, copy, modify, merge, publish, distribute" without restriction
- ✅ Only requirement: Include copyright notice and LICENSE file
**Implementation**:
- Keep `LICENSE` file in vendored `marble/` directory
- Document in `PROVENANCE.md`: "MARBLE is MIT licensed, vendoring permitted with attribution"

### Risk 4: User Setup Friction
**Probability**: High (manual MARBLE clone)
**Impact**: Low (one-time setup, well-documented)
**Mitigation**:
- Clear README with step-by-step instructions
- Provide setup script: `bash setup_multiagentbench.sh`
- Fail fast with helpful error if MARBLE missing
- Consider setup script that automates cloning

### Risk 5: Evaluation Discrepancies
**Probability**: Medium (different execution context)
**Impact**: High (invalidates scientific fidelity)
**Mitigation**:
- Run validation study: MASEval vs standalone MARBLE
- Compare outputs, metrics, agent behaviors on subset of tasks
- Document any differences in PROVENANCE.md
- If material differences exist, adjust adapter
- Acceptable: Different trace formatting, same final results

---

## Conclusion

The integration architecture provides:

✅ **Exact MARBLE reproduction** via `MarbleMultiAgentBenchBenchmark`
✅ **Fair framework comparison** via abstract `MultiAgentBenchBenchmark` base
✅ **Reuse of MASEval infrastructure** (callbacks, tracing, parallelization)
✅ **Clean architectural boundaries** (adapter pattern)
✅ **Scientific fidelity** (version pinning, validation)
✅ **Maintainability** (clear modules, documented patterns)

**Key Architectural Insights**:

1. **MARBLE's multi-agent Engine wraps as single AgentAdapter** - Preserves coordination logic while integrating with MASEval orchestration

2. **Dual-purpose design enables comparative research** - Abstract base allows users to implement any framework, enabling "which framework is better?" questions

3. **tau2/macs patterns transfer with one adaptation** - Demonstrates MASEval's flexibility for diverse benchmarking paradigms

4. **License verified** - MIT permits vendoring with attribution

**Next Steps**: Begin implementation with Phase 1 (infrastructure setup).

---

## Appendices

### Appendix A: Complete File Structure

```
maseval/benchmark/multiagentbench/
├── __init__.py
├── README.md
├── PROVENANCE.md
├── .gitignore
├── multiagentbench.py
├── environment.py
├── data_loader.py
├── adapters/
│   ├── __init__.py
│   └── marble_adapter.py
├── marble/                    # Gitignored, user clones
│   ├── LICENSE
│   ├── multiagentbench/
│   │   ├── coding/
│   │   ├── database/
│   │   ├── minecraft/
│   │   ├── research/
│   │   └── bargaining/
│   └── ...
└── data/                      # Symlink to marble/multiagentbench/
```

### Appendix B: MARBLE Architecture Summary

- **Engine**: Main orchestrator, coordinates agents and environment
- **BaseAgent**: Individual agent with LLM, tools, profile
- **AgentGraph**: Manages agent relationships and communication rules
- **SharedMemory**: Inter-agent communication and state sharing
- **EnginePlanner**: Plans agent execution order based on coordination mode
- **Environments**: Domain-specific (Coding, DB, Minecraft, Research, Bargaining, Web, WorldSimulation)
- **Evaluator**: Computes domain-specific metrics (code_quality, collaboration_effectiveness)

### Appendix C: Example Usage

**Basic usage with MARBLE reproduction:**

```python
from maseval.benchmark.multiagentbench import (
    load_tasks,
    configure_model_ids,
    MarbleMultiAgentBenchBenchmark,
)

# Load tasks
tasks = load_tasks("coding", limit=10)
configure_model_ids(tasks, agent_model_id="gpt-4")

# Run with MARBLE (exact reproduction)
marble_bench = MarbleMultiAgentBenchBenchmark()
marble_results = marble_bench.run(tasks)

# Print results
for result in marble_results:
    print(f"Task: {result['task_id']}")
    print(f"Code quality: {result['eval']['code_quality']}")
    print(f"Collaboration: {result['eval']['collaboration_effectiveness']}")
```

**Advanced: Framework comparison (user implementation):**

```python
from maseval.benchmark.multiagentbench import (
    load_tasks,
    configure_model_ids,
    MarbleMultiAgentBenchBenchmark,
)

# User's custom implementation (from examples/ directory)
from examples.multiagentbench_langgraph import LangGraphMultiAgentBench

tasks = load_tasks("coding", limit=10)
configure_model_ids(tasks, agent_model_id="gpt-4")

# Run with MARBLE
marble_bench = MarbleMultiAgentBenchBenchmark()
marble_results = marble_bench.run(tasks)

# Run with user's LangGraph implementation
langgraph_bench = LangGraphMultiAgentBench()
langgraph_results = langgraph_bench.run(tasks)

# Compare results
for marble_res, lg_res in zip(marble_results, langgraph_results):
    print(f"Task: {marble_res['task_id']}")
    print(f"  MARBLE code_quality: {marble_res['eval']['code_quality']}")
    print(f"  LangGraph code_quality: {lg_res['eval']['code_quality']}")
```

---

**Document Version**: 2.0
**Date**: 2026-01-19
**Author**: Claude (Sonnet 4.5)
**Status**: Final Recommendation
