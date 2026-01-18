# MultiAgentBench Integration Strategy

This document outlines strategies for integrating MultiAgentBench (MARBLE) into MASEval. It provides background context, analyzes three integration approaches, and recommends an implementation path.

## Table of Contents

1. [Background](#background)
2. [Integration Approaches](#integration-approaches)
3. [Recommendation](#recommendation)
4. [Implementation Roadmap](#implementation-roadmap)

---

## Background

### What is MultiAgentBench?

MultiAgentBench (implemented through the MARBLE framework) is a comprehensive benchmark for evaluating LLM-based multi-agent systems across diverse, interactive scenarios. Key characteristics:

- **6 Scenarios**: Research Collaboration, Minecraft Building, Database Anomaly Analysis, Coding Challenges, Werewolf Game, Bargaining
- **Dual Metrics**: Task completion (KPI, task score) + Coordination quality (communication, planning)
- **Multi-Agent Focus**: Unlike single-agent benchmarks, tests collaboration, competition, and emergent behaviors
- **Coordination Protocols**: Star, Chain, Tree, Graph-mesh topologies
- **Planning Strategies**: Vanilla, CoT, Group Discussion, Cognitive Self-Evolving

### Source Materials

- **Repository**: https://github.com/ulab-uiuc/MARBLE
- **License**: MIT License (Copyright 2024 Haofei Yu)
- **Paper**: Available in repository's `paper/` directory

### MARBLE Architecture Overview

```
MARBLE Framework
├── marble/                     # Core engine
│   ├── agent/                 # Agent implementations
│   ├── configs/               # YAML configuration
│   ├── engine/                # Simulation engine
│   ├── environments/          # Domain environments (Research, Coding, DB, Minecraft, Werewolf, Bargaining)
│   ├── evaluator/             # Metrics and evaluation
│   ├── graph/                 # Agent relationship graphs
│   ├── llms/                  # LLM integration (uses litellm)
│   ├── memory/                # Shared/individual memory
│   └── main.py               # Entry point
├── multiagentbench/           # Benchmark datasets (JSONL + converter)
└── scripts/                   # Domain-specific runners
```

### MASEval Architecture (for context)

```
maseval/
├── core/                      # Framework-agnostic abstractions
│   ├── benchmark.py          # Benchmark base class
│   ├── environment.py        # Environment base
│   ├── agent.py              # AgentAdapter (wraps external frameworks)
│   ├── evaluator.py          # Evaluator base
│   ├── model.py              # ModelAdapter (LLM abstraction)
│   └── tracing.py            # TraceableMixin, ConfigurableMixin
├── interface/                 # External framework adapters
│   ├── agents/               # smolagents, langgraph, llamaindex
│   └── inference/            # OpenAI, Anthropic, Google, etc.
└── benchmark/                 # Benchmark implementations
    ├── macs/                 # LLM-simulated tools, multi-agent hierarchy
    └── tau2/                 # Real tools, deterministic evaluation
```

### Key Design Considerations

1. **Replication Fidelity**: Must closely replicate MARBLE's evaluation methodology to reproduce paper results
2. **Framework Agnosticism**: MASEval supports multiple agent frameworks (smolagents, langgraph)
3. **Tool Execution Model**: MARBLE uses LLM-based tool simulation (similar to MACS, not Tau2)
4. **Multi-Agent Coordination**: MARBLE has its own coordination engine; MASEval's `run_agents()` is more minimal
5. **Memory Management**: MARBLE has sophisticated shared/individual memory; MASEval leaves this to agent frameworks
6. **Evaluation Complexity**: MARBLE evaluates via LLM (communication/planning scores) + domain-specific metrics

---

## Integration Approaches

### Approach 1: Port MARBLE Code (Recommended)

**Description**: Copy relevant MARBLE code into `maseval/benchmark/multiagentbench/`, adapting it to use MASEval's abstractions where beneficial while preserving MARBLE's core logic.

**What to Port**:
- Environments (`marble/environments/`) → Adapt to extend MASEval's `Environment`
- Evaluator logic (`marble/evaluator/`) → Implement as MASEval `Evaluator` subclasses
- Dataset loading (`multiagentbench/`) → Create `data_loader.py` similar to MACS/Tau2
- Prompt templates (evaluation prompts) → Store in `prompt_templates/`
- Domain-specific configurations

**What NOT to Port** (use MASEval instead):
- LLM integration → Use MASEval's `ModelAdapter` (LiteLLM, OpenAI, etc.)
- Agent framework wrappers → Use MASEval's `AgentAdapter` for smolagents/langgraph
- Tracing infrastructure → Use MASEval's `TraceableMixin`, `ConfigurableMixin`
- Configuration loading → Use MASEval's `Task`, `TaskQueue` patterns

**Structure**:
```
maseval/benchmark/multiagentbench/
├── __init__.py                    # Public API
├── multiagentbench.py            # MultiAgentBenchmark class(es)
├── data_loader.py                # Task loading, data management
├── environments/
│   ├── base_env.py               # BaseMultiAgentEnvironment
│   ├── research_env.py           # Research collaboration
│   ├── minecraft_env.py          # Minecraft building
│   ├── database_env.py           # Database anomaly analysis
│   ├── coding_env.py             # Coding challenges
│   ├── werewolf_env.py           # Werewolf game
│   └── bargaining_env.py         # Bargaining scenarios
├── evaluator.py                  # KPI, Task Score, Coordination Score
├── coordination/                 # Agent coordination protocols
│   ├── star.py
│   ├── chain.py
│   ├── tree.py
│   └── graph_mesh.py
├── prompt_templates/             # LLM evaluation prompts
│   ├── communication_eval.txt
│   ├── planning_eval.txt
│   └── task_eval/               # Per-domain task evaluation
└── data/                        # Downloaded benchmark data
```

**Pros**:
- Full control over integration
- Can optimize for MASEval patterns
- No external dependency management
- Easier debugging and modification
- Clear licensing (MIT allows copying with attribution)
- Can selectively port scenarios (start with 2-3, expand later)

**Cons**:
- Significant initial porting effort (~2-4 weeks)
- Must maintain parity with upstream changes
- Risk of subtle divergence from paper methodology
- Need deep understanding of MARBLE internals

**Effort Estimate**: High initial, low ongoing

---

### Approach 2: Install as Dependency

**Description**: Add `marble` as an optional dependency in `pyproject.toml` and create thin adapter classes that delegate to MARBLE.

**Implementation**:
```python
# pyproject.toml
[project.optional-dependencies]
multiagentbench = ["marble @ git+https://github.com/ulab-uiuc/MARBLE.git"]

# maseval/benchmark/multiagentbench/multiagentbench.py
from maseval import Benchmark, Environment, Task

class MultiAgentBenchmark(Benchmark):
    def setup_environment(self, agent_data, task):
        # Import MARBLE at runtime
        from marble.environments import get_environment
        from marble.configs import Config

        # Create MARBLE environment, wrap for MASEval
        marble_config = Config.from_dict(task.environment_data)
        marble_env = get_environment(marble_config)
        return MultiAgentBenchEnvironmentWrapper(marble_env)

    def run_agents(self, agents, task, environment, query):
        # Delegate to MARBLE's coordination engine
        from marble.engine import Engine
        engine = Engine(environment.marble_env)
        return engine.run()
```

**Pros**:
- Minimal code to write
- Automatically gets upstream bug fixes
- Guaranteed methodology parity
- Faster initial implementation (~1 week)

**Cons**:
- **Heavy dependency**: MARBLE has many dependencies (litellm, beautifulsoup4, flask, psycopg2, etc.)
- **Version pinning complexity**: Must track MARBLE versions for reproducibility
- **API surface mismatch**: MARBLE's coordination engine doesn't fit MASEval's `run_agents()` model cleanly
- **Framework conflict**: Both use litellm differently; potential version conflicts
- **Installation friction**: Users must install from git URL (not PyPI)
- **Limited customization**: Hard to use MASEval's smolagents/langgraph adapters

**Effort Estimate**: Low initial, medium ongoing (dependency management)

---

### Approach 3: Hybrid Dynamic Import

**Description**: Port the essential components (environments, evaluation) but dynamically load MARBLE's coordination engine when available, falling back to a simplified implementation otherwise.

**Implementation**:
```python
# maseval/benchmark/multiagentbench/coordination.py

def get_coordination_engine(protocol: str):
    """Get coordination engine, preferring MARBLE if available."""
    try:
        from marble.engine import EnginePlanner
        from marble.graph import AgentGraph
        return MARBLECoordinationAdapter(protocol)
    except ImportError:
        # Fall back to simplified implementation
        return SimpleCoordinationEngine(protocol)

class SimpleCoordinationEngine:
    """Simplified coordination for basic usage without full MARBLE."""

    def __init__(self, protocol: str):
        self.protocol = protocol

    def coordinate(self, agents, task, environment):
        if self.protocol == "star":
            return self._star_coordinate(agents, task, environment)
        elif self.protocol == "chain":
            return self._chain_coordinate(agents, task, environment)
        # ... etc
```

**Structure**:
```
maseval/benchmark/multiagentbench/
├── multiagentbench.py            # Benchmark class
├── data_loader.py                # Task loading (ported)
├── environments/                 # Ported from MARBLE
├── evaluator.py                  # Ported from MARBLE
├── coordination/
│   ├── __init__.py              # get_coordination_engine()
│   ├── simple.py                # Fallback implementations
│   └── marble_adapter.py        # MARBLE delegation (if available)
└── data/
```

**Pros**:
- Works with or without MARBLE installed
- Can start simple, add MARBLE compatibility later
- Best of both worlds for evaluation fidelity
- Users who don't need full coordination can use lightweight mode

**Cons**:
- Two code paths to maintain
- Potential behavior differences between modes
- More complex testing (need to test both paths)
- "Simplified" coordination may not replicate paper results accurately

**Effort Estimate**: Medium initial, high ongoing (maintaining two implementations)

---

## Recommendation

**Recommended Approach: Port MARBLE Code (Approach 1)**

### Rationale

1. **Replication Accuracy**: The primary goal is to replicate paper results. Porting gives full control over the evaluation methodology and ensures we understand every detail of how metrics are computed.

2. **MASEval Alignment**: MARBLE's architecture differs significantly from MASEval's patterns:
   - MARBLE has its own coordination engine → We can adapt this to work with MASEval's `run_agents()`
   - MARBLE uses litellm directly → We can use MASEval's `ModelAdapter` for unified provider support
   - MARBLE has YAML configs → We can convert to MASEval's `Task` objects

3. **Dependency Management**: MARBLE has 15+ dependencies, some with specific version requirements. Adding it as a dependency creates potential conflicts with MASEval's existing dependencies.

4. **Framework Flexibility**: Porting allows us to properly support MASEval's agent framework adapters (smolagents, langgraph), rather than being locked to MARBLE's agent implementation.

5. **MIT License**: MARBLE's MIT license explicitly permits copying, modification, and redistribution with attribution.

6. **MACS Precedent**: The MACS benchmark followed a similar pattern—porting core logic while adapting to MASEval's abstractions. This approach has proven successful.

### Phased Implementation

Rather than porting all 6 scenarios at once, implement in phases:

**Phase 1 (MVP)**: Research Collaboration + Bargaining
- Both are text-only, no external services needed
- Research tests collaborative task completion
- Bargaining tests competitive interaction
- Validates core infrastructure (agent graph, coordination, evaluation)

**Phase 2**: Coding + Database
- More complex tool usage
- Tests multi-role collaboration (debugger, reviewer, etc.)
- Deterministic evaluation possible for coding (execution success)

**Phase 3**: Minecraft + Werewolf
- Minecraft requires understanding block placement semantics
- Werewolf requires complex game state management
- Most complex coordination patterns

---

## Implementation Roadmap

### Prerequisites

- [ ] Confirm MARBLE repository access and license compliance
- [ ] Review MARBLE test suite for validation criteria
- [ ] Identify minimal set of MARBLE dependencies to preserve

### Phase 1: Foundation (Research + Bargaining)

**Week 1: Infrastructure**
- [ ] Create `maseval/benchmark/multiagentbench/` directory structure
- [ ] Port data loading (`data_loader.py`) with GitHub download support
- [ ] Implement base environment class
- [ ] Create Task/TaskQueue conversion from MARBLE JSONL format

**Week 2: Environments**
- [ ] Port ResearchEnvironment with tools (paper fetching, etc.)
- [ ] Port BargainingEnvironment with negotiation tools
- [ ] Adapt tools to work with MASEval's tracing infrastructure

**Week 3: Coordination & Agents**
- [ ] Implement Agent Graph construction from configs
- [ ] Implement Star and Graph-mesh coordination protocols
- [ ] Create `MultiAgentBenchmark` class with `setup_agents()`, `run_agents()`
- [ ] Framework-agnostic agent wrapper (similar to MACSGenericTool pattern)

**Week 4: Evaluation**
- [ ] Port LLM-based communication score evaluation
- [ ] Port LLM-based planning score evaluation
- [ ] Implement domain-specific task scores (Research: innovation/safety/feasibility, Bargaining: negotiation metrics)
- [ ] Port Milestone-based KPI calculation
- [ ] Create `compute_benchmark_metrics()` aggregation function

**Week 5: Testing & Validation**
- [ ] Write unit tests for environments, evaluation
- [ ] Run baseline tests with GPT-4o-mini
- [ ] Compare results to paper Table 1 values
- [ ] Document any deviations and their causes

### Phase 2: Coding + Database (TBD after Phase 1)

### Phase 3: Minecraft + Werewolf (TBD after Phase 2)

---

## Key Implementation Details

### Agent Coordination Pattern

MARBLE's coordination differs from MASEval's single-agent assumption. The recommended pattern:

```python
class MultiAgentBenchmark(Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        """Build agent graph from task configuration."""
        agent_configs = task.environment_data.get("agents", [])
        coordination_mode = task.environment_data.get("coordinate_mode", "graph")

        # Build agents with relationships
        agents = self._build_agent_graph(agent_configs, environment)

        # Store coordination mode for run_agents
        self._coordination_mode = coordination_mode

        # Return primary agent(s) for execution
        primary_agents = [a for a in agents if a.is_primary]
        agents_dict = {a.name: a for a in agents}

        return primary_agents, agents_dict

    def run_agents(self, agents, task, environment, query):
        """Execute agents using appropriate coordination protocol."""
        if self._coordination_mode == "star":
            return self._run_star(agents, query, environment)
        elif self._coordination_mode == "graph":
            return self._run_graph_mesh(agents, query, environment)
        # ... etc
```

### Evaluation Metrics Structure

```python
@dataclass
class MultiAgentBenchEvalResult:
    # Task completion
    task_score: float        # Domain-specific (0-5 scale, normalized)
    kpi: float               # Milestone-based KPI (0-1)

    # Coordination quality
    communication_score: float   # LLM-evaluated (1-5 scale)
    planning_score: float        # LLM-evaluated (1-5 scale)
    coordination_score: float    # Average of above (1-5 scale)

    # Per-agent breakdown
    agent_kpis: Dict[str, float]  # Individual agent KPIs

    # Raw data
    milestone_report: List[Dict]  # Milestone tracking details
    eval_report: Dict             # Full LLM evaluation outputs
```

### Data Loading Pattern

```python
def load_tasks(
    scenario: str,  # "research", "bargaining", "coding", etc.
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> TaskQueue:
    """Load MultiAgentBench tasks for a scenario."""

    ensure_data_exists(scenario, data_dir)

    # Load JSONL and convert
    raw_tasks = _load_jsonl(data_dir / scenario / "tasks.jsonl")

    tasks = []
    for raw in raw_tasks:
        task = Task(
            id=raw.get("task_id"),
            query=raw["task"]["content"],
            environment_data={
                "scenario": scenario,
                "agents": raw["agents"],
                "relationships": raw["relationships"],
                "coordinate_mode": raw.get("coordinate_mode", "graph"),
                "tools": raw.get("environment", {}).get("tools", []),
            },
            evaluation_data={
                "milestones": raw.get("milestones", []),
                "task_rubric": raw.get("evaluation", {}),
            },
            metadata={
                "scenario": scenario,
                "difficulty": raw.get("difficulty", "medium"),
            },
        )
        tasks.append(task)

    return TaskQueue(tasks[:limit] if limit else tasks)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Subtle methodology differences causing result divergence | Carefully validate against paper results; document any deviations |
| MARBLE updates breaking compatibility | Pin to specific commit hash; periodically review upstream changes |
| LLM evaluation non-determinism | Use fixed seeds where possible; report variance across runs |
| Minecraft/Werewolf complexity | Defer to Phase 3; may require simplified versions initially |
| Token cost for LLM-based evaluation | Provide cost estimates in documentation; support cheaper evaluation models |

---

## License Compliance

MARBLE is released under the MIT License. To comply:

1. Include MIT license text in `maseval/benchmark/multiagentbench/LICENSE`
2. Add attribution in module docstrings:
   ```python
   """
   MultiAgentBench integration for MASEval.

   Based on MARBLE: https://github.com/ulab-uiuc/MARBLE
   Copyright (c) 2024 Haofei Yu
   Licensed under the MIT License
   """
   ```
3. Note original authors in CHANGELOG when adding the benchmark

---

## Conclusion

Porting MARBLE code (Approach 1) provides the best balance of replication fidelity, MASEval integration, and long-term maintainability. The phased implementation approach allows incremental validation while managing complexity.

Starting with Research Collaboration and Bargaining scenarios provides a solid foundation that exercises all key components (agent graphs, coordination, LLM evaluation) while avoiding the complexity of Minecraft's spatial reasoning or Werewolf's game state management.
