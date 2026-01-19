# STRATEGY.md Critique: MultiAgentBench Integration

This document analyzes the proposed STRATEGY.md for integrating MARBLE/MultiAgentBench into MASEval, comparing against established patterns from tau2 and macs benchmarks.

---

## Executive Summary

The STRATEGY.md proposes a reasonable high-level architecture but contains several issues:

| Category | Count | Severity |
|----------|-------|----------|
| **Incorrect API Assumptions** | 5 | High |
| **Pattern Violations** | 3 | Medium |
| **Clumsy Implementations** | 4 | Low-Medium |
| **Missing Considerations** | 3 | Medium |

**Overall Assessment**: The strategy requires significant revision before implementation, particularly around MARBLE API assumptions and the "Engine as single adapter" pattern.

---

## 1. Incorrect MARBLE API Assumptions

### 1.1 SharedMemory is NOT Actually Shared (HIGH SEVERITY)

**STRATEGY.md claims:**
```python
# Extract final answer from SharedMemory or designated output
final_answer = self.engine.memory.get("final_answer")
```

**Reality**: Each MARBLE agent creates its own `SharedMemory` instance at initialization (`base_agent.py:73`). The memory is NOT shared between agents despite the misleading name. There is no central `engine.memory` that accumulates agent outputs.

**Evidence from MARBLE source:**
```python
# In BaseAgent.__init__():
self.memory = BaseMemory()  # Per-agent memory
self.shared_memory = SharedMemory()  # Also per-agent, NOT shared!
```

**Impact**: The proposed `_convert_to_message_history()` and final answer extraction logic will fail silently or return empty data.

**Recommendation**: Extract final answers from:
- `engine.agents[-1].last_output` (if exists)
- Agent task_history or msg_box
- Environment state after execution

---

### 1.2 Missing `get_agent_profiles_linked()` Method (HIGH SEVERITY)

**STRATEGY.md implies MARBLE's chain coordination works:**
> "Agents use AgentGraph to check who they can communicate with"

**Reality**: `engine.py:702` calls `self.graph.get_agent_profiles_linked(current_agent.agent_id)` but this method **does not exist** in `AgentGraph`. This is a bug in MARBLE that will crash chain coordination mode.

**Impact**: Cannot reliably run tasks with `coordinate_mode: "chain"` using vanilla MARBLE.

**Recommendation**:
1. Document this limitation
2. Either patch MARBLE locally or avoid chain mode tasks initially
3. File upstream issue

---

### 1.3 Message History Access Pattern is Wrong (HIGH SEVERITY)

**STRATEGY.md proposes:**
```python
def _convert_to_message_history(self, engine) -> MessageHistory:
    for agent in engine.agents:
        if hasattr(agent, "messages"):
            for msg in agent.messages:
                # ...
```

**Reality**: MARBLE agents store messages in `msg_box`, not a `messages` attribute:
```python
# BaseAgent structure:
self.msg_box: Dict[session_id, Dict[agent_id, List[Tuple[direction, message]]]]
# Direction: 0 = FORWARD_TO (sent), 1 = RECV_FROM (received)
```

Messages are serialized via `agent.seralize_message(session_id)` (note: typo in MARBLE).

**Recommendation**: Use the actual MARBLE API:
```python
for agent in engine.agents:
    serialized = agent.seralize_message("")  # Empty session_id for all
    # Parse serialized string or access msg_box directly
```

---

### 1.4 `engine.start()` Return Value Not Specified

**STRATEGY.md assumes:**
```python
def _run_agent(self, query: str) -> Any:
    self.engine.start()
    # ... extract results
```

**Reality**: `engine.start()` dispatches to coordination methods (`star_coordinate`, `chain_coordinate`, etc.) but the return behavior varies and is not consistently documented. Some modes return results, others mutate state.

**Recommendation**: Study each coordination mode's return behavior and document clearly. Don't assume uniform behavior.

---

### 1.5 Evaluator.update() Must Be Called Explicitly

**STRATEGY.md implies:**
> "MARBLE evaluator metrics (code_quality, collaboration)"

**Reality**: MARBLE's `Evaluator.update(environment, agents)` must be called explicitly after each iteration—the engine does NOT call it automatically. Without explicit calls, metrics remain empty.

**Impact**: `MultiAgentBenchEvaluator` wrapper may return empty metrics.

**Recommendation**: Call `marble_evaluator.update()` after engine execution, then `marble_evaluator.finalize()` before reading metrics.

---

## 2. Pattern Violations

### 2.1 "Engine as Single AgentAdapter" Violates Multi-Agent Visibility (MEDIUM SEVERITY)

**STRATEGY.md proposes:**
```python
# Wrap entire engine as single adapter
engine_adapter = MarbleAgentAdapter(engine, "marble_engine")
return [engine_adapter], {"marble_engine": engine_adapter}
```

**MASEval Pattern (from tau2/macs)**:
- Each agent gets its own `AgentAdapter`
- `agents_dict` maps agent names to individual adapters
- Traces are collected per-agent for analysis

**Why this matters**:
1. **Trace granularity lost**: Cannot analyze individual agent behavior
2. **Callback hooks incomplete**: `on_agent_step_start/end` fires once for entire system
3. **Error attribution unclear**: Which internal agent failed?
4. **Inconsistent with macs**: MACS explicitly creates adapters per agent in hierarchy

**Comparison to MACS pattern:**
```python
# MACS approach - each agent is visible
def setup_agents(self, agent_data, environment, task, user):
    adapters = []
    agents_dict = {}
    for agent_spec in agent_data["agents"]:
        agent = create_agent(agent_spec, environment)
        adapter = AgentAdapter(agent, agent_spec["id"])
        adapters.append(adapter)
        agents_dict[agent_spec["id"]] = adapter
    return adapters, agents_dict
```

**Justified Reason to Deviate**: MARBLE's coordination logic is tightly coupled in the Engine. Exposing individual agents would require reimplementing coordination, defeating the purpose of scientific fidelity.

**Recommendation**: Document this as an explicit architectural trade-off. Consider:
1. Keep Engine-as-adapter for `MarbleMultiAgentBenchBenchmark` (reproduction mode)
2. For custom framework implementations, encourage per-agent adapters
3. Enhance `gather_traces()` to expose internal agent details as nested structure

---

### 2.2 Environment.create_tools() Returns Empty Dict (MEDIUM SEVERITY)

**STRATEGY.md proposes:**
```python
def create_tools(self) -> Dict[str, Any]:
    # MARBLE environments manage tools internally
    return {}

def get_tool(self, name: str) -> Optional[Any]:
    raise NotImplementedError("Tools managed by MARBLE Engine")
```

**MASEval Pattern (from tau2/macs)**:
- `Environment.create_tools()` returns actual tool instances
- Tools are registered and traced via MASEval infrastructure
- Agents receive tools from environment

**tau2 example:**
```python
def create_tools(self) -> Dict[str, Callable]:
    return self.toolkit.get_tools()  # Actual callable tools
```

**macs example:**
```python
def create_tools(self) -> Dict[str, MACSGenericTool]:
    tools = {}
    for spec in self.state["tool_specs"]:
        tools[spec["name"]] = MACSGenericTool(spec, model)
    return tools
```

**Why this matters**:
1. **Tool tracing lost**: MASEval won't capture tool invocations
2. **ToolInvocationHistory empty**: No tool usage data in traces
3. **Inconsistent behavior**: Users expect `environment.tools` to work

**Recommendation**: Either:
1. Extract tool definitions from MARBLE and create MASEval-traceable wrappers
2. Or document clearly that tool tracing requires post-hoc extraction from engine traces

---

### 2.3 User Simulator Returning None (MINOR)

**STRATEGY.md proposes:**
```python
def setup_user(self, agent_data, environment, task, user):
    # → None (MultiAgentBench doesn't use external user simulation)
    return None
```

**This is acceptable** - tau2 and macs support optional user simulation. However, MARBLE does have some scenarios with user interaction.

**Recommendation**: Verify which MultiAgentBench domains require user simulation and handle appropriately.

---

## 3. Clumsy Implementations

### 3.1 Overly Complex Config Conversion

**STRATEGY.md proposes:**
```python
marble_config = Config.from_dict({
    "coordinate_mode": task.environment_data["coordinate_mode"],
    "relationships": task.environment_data["relationships"],
    "agents": task.environment_data["agents"],
    "environment": task.environment_data["environment"],
    "task": {"content": task.query},
    "llm": task.environment_data["llm"],
    "memory": {"type": "SharedMemory"},
    "engine_planner": {"initial_progress": "Starting task"},
    "metrics": task.evaluation_data["metrics"],
})
```

**Issue**: This duplicates MARBLE's config loading. MARBLE already has `Config.from_file()` and the JSONL entries ARE MARBLE configs.

**Cleaner approach:**
```python
# MARBLE configs are stored directly in environment_data
marble_config = Config.from_dict(task.environment_data["raw_marble_config"])
```

Or even simpler - load MARBLE tasks directly and convert to MASEval Task:
```python
def load_tasks(domain):
    # Let MARBLE parse its own format
    marble_tasks = marble.load_tasks(domain)
    return [Task.from_marble(mt) for mt in marble_tasks]
```

---

### 3.2 Symlink for Data Directory is Fragile

**STRATEGY.md proposes:**
```
└── data/                      # Symlink to marble/multiagentbench/
```

**Issues**:
1. Symlinks don't work well on Windows
2. Adds complexity to setup
3. Git doesn't track symlinks reliably

**Cleaner approach (tau2 pattern):**
```python
# In data_loader.py
def _get_data_dir() -> Path:
    # Look for MARBLE in known locations
    candidates = [
        Path(__file__).parent / "marble" / "multiagentbench",
        Path(os.environ.get("MARBLE_DATA_DIR", "")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("MARBLE data not found. See README.md for setup.")
```

---

### 3.3 Redundant Fallback Strategies for Message History

**STRATEGY.md proposes three fallback strategies:**
```python
# Strategy 1: Direct message access
# Strategy 2: Fallback to SharedMemory
# Strategy 3: Fallback to agent outputs
```

**Issue**: Strategy 2 won't work (SharedMemory isn't shared). This creates dead code and false confidence.

**Recommendation**: Remove non-functional fallbacks. Be explicit about what actually works:
```python
def _extract_messages(self, engine) -> List[Dict]:
    """Extract messages from MARBLE agents.

    Note: MARBLE stores messages in agent.msg_box, not a central location.
    """
    messages = []
    for agent in engine.agents:
        # Only method that actually works
        for session_id, conversations in agent.msg_box.items():
            for other_agent_id, msg_list in conversations.items():
                for direction, content in msg_list:
                    messages.append({
                        "agent_id": agent.agent_id,
                        "direction": "sent" if direction == 0 else "received",
                        "to/from": other_agent_id,
                        "content": content,
                    })
    return messages
```

---

### 3.4 Over-Specification in Implementation Checklist

The checklist has 40+ items spanning 7 phases. Compare to tau2 implementation which is ~500 lines across 4 files.

**Recommendation**: Simplify to MVP:
1. Phase 1: Data loader + basic environment
2. Phase 2: MarbleAgentAdapter + benchmark class
3. Phase 3: Evaluator integration
4. Phase 4: Tests + example

Defer user framework examples (LangGraph, smolagents) to later PRs.

---

## 4. Missing Considerations

### 4.1 No Discussion of Domain-Specific Environments

**MARBLE has 7 distinct environments:**
- CodingEnvironment (file operations)
- DBEnvironment (Docker + PostgreSQL)
- MinecraftEnvironment (external process)
- ResearchEnvironment
- BargainingEnvironment
- WebEnvironment
- WorldSimulationEnvironment

**Each has different setup requirements:**
- DBEnvironment requires Docker
- MinecraftEnvironment requires external game server
- CodingEnvironment needs filesystem access

**STRATEGY.md mentions this but doesn't address:**
- How to handle missing dependencies gracefully
- Which domains to support initially
- How to skip domains without required infrastructure

**Recommendation**: Start with simpler domains (Research, Bargaining) that don't require external services. Add infrastructure-heavy domains (DB, Minecraft) as optional extras.

---

### 4.2 No Error Classification Strategy

**MASEval uses structured error attribution:**
```python
class TaskExecutionStatus(Enum):
    SUCCESS = "success"
    AGENT_ERROR = "agent_error"
    ENVIRONMENT_ERROR = "environment_error"
    USER_ERROR = "user_error"
    TASK_TIMEOUT = "task_timeout"
    # ...
```

**STRATEGY.md doesn't discuss:**
- How MARBLE errors map to these statuses
- What happens when MARBLE's internal agents fail
- How to distinguish MARBLE bugs from agent failures

**Recommendation**: Add error classification logic:
```python
def _classify_marble_error(self, error: Exception) -> TaskExecutionStatus:
    if "MARBLE" in str(error) or isinstance(error, MarbleInternalError):
        return TaskExecutionStatus.ENVIRONMENT_ERROR  # MARBLE bug
    if "LLM" in str(error) or "rate limit" in str(error).lower():
        return TaskExecutionStatus.ENVIRONMENT_ERROR  # External service
    return TaskExecutionStatus.AGENT_ERROR  # Agent's fault
```

---

### 4.3 Evaluation Metrics Not Fully Mapped

**MARBLE Evaluator produces:**
- `task_completion`: List[0/1]
- `token_consumption`: List[int]
- `planning_score`: List[1-5]
- `communication_score`: List[1-5]
- `code_quality`: Dict
- `agent_kpis`: Dict[agent_id, milestone_count]

**MASEval evaluation format:**
```python
{"metric_name": value, "passed": bool, ...}
```

**STRATEGY.md proposes:**
```python
return self.marble_evaluator.evaluate(output, task_spec, traces)
```

But doesn't specify how MARBLE's metrics map to MASEval's format.

**Recommendation**: Define explicit mapping:
```python
def __call__(self, traces, final_answer) -> Dict[str, Any]:
    marble_metrics = self.marble_evaluator.get_metrics()
    return {
        "passed": marble_metrics.get("task_completion", [0])[-1] == 1,
        "task_completion": marble_metrics.get("task_completion", [0])[-1],
        "planning_score": mean(marble_metrics.get("planning_score", [3])),
        "communication_score": mean(marble_metrics.get("communication_score", [3])),
        "total_tokens": sum(marble_metrics.get("token_consumption", [0])),
        # ...
    }
```

---

## 5. Justified Pattern Deviations

### 5.1 Vendoring MARBLE Source (JUSTIFIED)

**Deviation**: Vendoring entire MARBLE repo rather than pip dependency.

**Justification**:
- MARBLE's pip package has bugs
- Need to pin exact version for reproducibility
- May need local patches
- MIT license explicitly permits this

**This is the right choice** - tau2 does something similar with its data.

---

### 5.2 Abstract Base + Concrete MARBLE Implementation (JUSTIFIED)

**Deviation**: Creating `MultiAgentBenchBenchmark` (abstract) and `MarbleMultiAgentBenchBenchmark` (concrete).

**Justification**:
- Enables framework comparison research
- Matches MASEval's "BYO agent" philosophy
- Same pattern used by tau2 and macs

**This is well-aligned** with MASEval patterns.

---

### 5.3 Engine-Level Wrapping for Scientific Fidelity (PARTIALLY JUSTIFIED)

**Deviation**: Wrapping MARBLE Engine as single adapter.

**Partial Justification**:
- Preserves MARBLE's exact coordination logic
- Required for reproducing published results
- Would require reimplementing MARBLE to do otherwise

**Concern**: Loses per-agent observability. Should be documented as explicit trade-off with enhanced trace extraction to compensate.

---

## 6. Recommended Changes Before Implementation

### High Priority (Must Fix)

1. **Fix SharedMemory assumptions** - It's not shared; update all code that assumes it is
2. **Fix message history extraction** - Use `msg_box` and `seralize_message()`
3. **Document chain mode bug** - `get_agent_profiles_linked()` doesn't exist
4. **Add explicit Evaluator.update() calls**

### Medium Priority (Should Fix)

5. **Enhance traces from Engine adapter** - Include internal agent details as nested structure
6. **Simplify config conversion** - Use MARBLE's native loading where possible
7. **Add error classification** - Map MARBLE errors to MASEval statuses
8. **Define metric mapping** - MARBLE metrics → MASEval format

### Low Priority (Nice to Have)

9. **Remove symlink approach** - Use path resolution instead
10. **Simplify implementation phases** - Start with MVP
11. **Document domain requirements** - Which need Docker, external services, etc.
12. **Remove non-functional fallbacks** - Dead code creates false confidence

---

## 7. Conclusion

The STRATEGY.md demonstrates good understanding of MASEval's architecture but makes several incorrect assumptions about MARBLE's API that would cause runtime failures. The "Engine as single adapter" pattern is a reasonable trade-off for scientific fidelity but should be clearly documented with enhanced trace extraction.

**Recommended next steps:**
1. Revise STRATEGY.md with corrections from this critique
2. Create minimal proof-of-concept with single domain (Research or Bargaining)
3. Validate MARBLE API assumptions with actual execution
4. Expand to other domains incrementally

---

*Document Version: 1.0*
*Date: 2026-01-19*
*Reviewer: Claude (Opus 4.5)*
