# Tau 2 Benchmark Integration: Implementation Options

This document evaluates multiple architectural strategies for integrating the **Tau 2 benchmark** ([tau2-bench](https://github.com/sierra-research/tau2-bench)) into the MASEval library. Each strategy is analyzed against five key criteria to determine the optimal integration approach.

---

## Background

### Current State

- **MASEval**: MIT-licensed evaluation framework for multi-agent systems
- **tau2-bench**: MIT-licensed benchmark for evaluating agentic systems in multi-turn interactive environments
- **MACS Benchmark**: Reference implementation in MASEval—built from scratch using MASEval's core abstractions

### Key Differences from MACS

The MACS benchmark was implemented entirely from scratch because the original AWS repository only provided data and evaluation prompts. In contrast, **tau2-bench provides a complete, functional codebase** including:

- Domain implementations (airline, retail, telecom)
- Agent evaluation infrastructure
- User simulators
- Gymnasium-compatible environment interface
- CLI tooling and configuration management
- Pass^k metric computation

### tau2-bench Technical Profile

| Aspect       | Details                                                               |
| ------------ | --------------------------------------------------------------------- |
| License      | MIT (Copyright 2025 Sierra Research)                                  |
| Python       | >=3.10                                                                |
| Dependencies | 29 packages (litellm, gymnasium, FastAPI, pandas, scikit-learn, etc.) |
| Installation | `pip install` from Git URL (not on PyPI)                              |
| Entry Point  | `tau2` CLI command                                                    |

### Fundamental Difference in Focus

| Aspect              | tau2-bench                     | MASEval                                                        |
| ------------------- | ------------------------------ | -------------------------------------------------------------- |
| **Primary Goal**    | Model comparison               | System comparison                                              |
| **Evaluation Unit** | LLM performance on fixed tasks | Full agent system (architecture, prompts, tools, coordination) |
| **Use Case**        | "Which LLM performs best?"     | "Which agent architecture performs best?"                      |

This distinction is critical: tau2-bench's evaluation logic is optimized for comparing models, while MASEval needs flexibility to evaluate different system designs.

---

## MASEval's Existing Capabilities

Before evaluating integration strategies, it's essential to understand what MASEval **already provides** that overlaps with tau2-bench functionality:

### Multi-Turn Interaction (Already Implemented)

```python
# Benchmark class
class Benchmark:
    def __init__(self, ..., n_task_repeats=1, max_invocations=1):
        self.max_invocations = max_invocations  # Agent-user exchange rounds

    def execution_loop(self, agents, task, environment, user):
        """Built-in agent-user interaction loop with termination handling."""
        for _ in range(self.max_invocations):
            final_answer = self.run_agents(agents, task, environment, query_text)
            if user.is_done():
                break
            query_text = user.simulate_response(str(final_answer))
```

### User Simulation (Already Implemented)

```python
# User class with multi-turn support
class User:
    def __init__(self, ..., max_turns=1, stop_token=None, early_stopping_condition=None):
        self.simulator = UserLLMSimulator(...)  # LLM-based response generation

    def simulate_response(self, question: str) -> str:
        """Generate realistic user response via LLM."""

    def is_done(self) -> bool:
        """Check max_turns or stop_token termination."""
```

### Tool Simulation (Already Implemented)

```python
# ToolLLMSimulator in maseval/core/simulator.py
class ToolLLMSimulator:
    """Simulates tool responses using an LLM when real APIs aren't available."""
```

### Task Repetition for Statistical Metrics (Already Implemented)

```python
# Pass^k requires running each task k times - MASEval has this
benchmark = MyBenchmark(agent_data=config, n_task_repeats=4)
# Then aggregate: "did at least 1 of k runs succeed?"
```

### What tau2-bench Provides vs. What MASEval Already Has

| Component                              | tau2-bench          | MASEval Status                         |
| -------------------------------------- | ------------------- | -------------------------------------- |
| Domain data (airline, retail, telecom) | JSON config files   | **Need to download**                   |
| Policy constraints                     | Rule specifications | **Need to implement**                  |
| Tool definitions                       | JSON schemas        | **Need to parse**                      |
| Pass^k metric computation              | ~50 lines           | **Trivial with `n_task_repeats`**      |
| Multi-turn dialogue orchestration      | Custom orchestrator | **`execution_loop()` exists**          |
| User simulation                        | LLM-based           | **`UserLLMSimulator` exists**          |
| Tool simulation                        | LLM-based           | **`ToolLLMSimulator` exists**          |
| FastAPI server                         | Agent API endpoint  | **Not needed for MASEval**             |
| Gymnasium interface                    | RL training         | **Not needed for MASEval**             |
| Redis caching                          | LLM call cache      | **Not needed (MASEval has callbacks)** |

**Key Insight**: The only truly unique asset in tau2-bench is the **domain data and specifications**. The evaluation infrastructure largely duplicates what MASEval already provides.

---

## Strategy 1: Pip Install from Git URL (Optional Dependency)

### Description

Add tau2-bench as an optional dependency installed directly from its Git repository. Create a thin adapter layer in `maseval/benchmark/tau2/` that wraps tau2-bench's evaluation infrastructure.

```toml
# pyproject.toml
[project.optional-dependencies]
tau2 = ["tau2 @ git+https://github.com/sierra-research/tau2-bench.git"]
```

### Implementation Approach

1. Add tau2-bench as optional dependency from Git URL
2. Create `Tau2Benchmark` class that delegates to tau2-bench's evaluator
3. Implement `Tau2Environment` and `Tau2User` as thin wrappers
4. Map tau2-bench results to MASEval's result format

### Trade-off Analysis

#### 1. Licensing Compatibility

| Rating        | Analysis                                                                                                             |
| ------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Excellent** | Both projects use MIT license. No conflicts or special attribution requirements beyond preserving copyright notices. |

#### 2. Maintenance & Upstream Sync

| Rating   | Analysis                                                                                                                                                                                                                                 |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Good** | Updates are pulled automatically on reinstall. However, API changes in tau2-bench could break MASEval's adapter layer without warning. Version pinning (e.g., `tau2 @ git+...@v0.2.1`) mitigates this but requires manual version bumps. |

#### 3. Ease of Use / Installation

| Rating       | Analysis                                                                                                                                                                                                        |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Moderate** | Git-based dependencies require Git to be installed. Some corporate environments block Git-based pip installs. The 29 transitive dependencies add significant installation time and potential version conflicts. |

**Dependency Bloat**:

```
tau2-bench dependencies (29 packages):
- FastAPI, uvicorn (server - not needed)
- gymnasium (RL interface - not needed)
- pandas, scikit-learn (data science - not needed)
- plotly, seaborn, matplotlib (visualization - not needed)
- redis (caching - not needed)
- litellm (LLM abstraction - MASEval has ModelAdapter)
```

#### 4. Architectural Consistency

| Rating   | Analysis                                                                                                                                                                                                                      |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Poor** | Creates a dual-layer architecture unlike MACS. tau2-bench uses its own patterns (`registry.py`, custom orchestrator) that don't align with MASEval's class hierarchy. Callbacks and tracing would require translation layers. |

#### 5. Reproducibility

| Rating        | Analysis                                                                              |
| ------------- | ------------------------------------------------------------------------------------- |
| **Excellent** | Uses original evaluation code unchanged. Results identical to upstream by definition. |

---

## Strategy 2: Adapter Pattern with Selective Imports

### Description

Install tau2-bench as a dependency, but only import and use specific modules (evaluator, domains, data_model) rather than the full framework. Build MASEval-native components that internally delegate to tau2-bench's core logic.

### Implementation Approach

1. Install tau2-bench as optional dependency
2. Import only: `tau2.evaluator`, `tau2.domains`, `tau2.data_model`
3. Implement `Tau2Environment(Environment)` using tau2 domain definitions
4. Implement `Tau2Evaluator(Evaluator)` wrapping tau2's evaluator
5. Build `Tau2User(User)` using tau2's user simulation logic
6. Create `Tau2Benchmark(Benchmark)` orchestrating MASEval's execution flow

### Trade-off Analysis

#### 1. Licensing Compatibility

| Rating        | Analysis                                         |
| ------------- | ------------------------------------------------ |
| **Excellent** | Same as Strategy 1—MIT license fully compatible. |

#### 2. Maintenance & Upstream Sync

| Rating       | Analysis                                                                                                                                      |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Moderate** | Selective imports create coupling to specific internal APIs. Changes to tau2's internal module structure could break imports without warning. |

#### 3. Ease of Use / Installation

| Rating       | Analysis                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------ |
| **Moderate** | Same dependency bloat as Strategy 1—all 29 packages still installed even if only using a subset. |

#### 4. Architectural Consistency

| Rating       | Analysis                                                                                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Moderate** | Better than Strategy 1 since MASEval controls execution flow. However, still dealing with impedance mismatch between tau2's internal patterns and MASEval's abstractions. |

#### 5. Reproducibility

| Rating        | Analysis                         |
| ------------- | -------------------------------- |
| **Excellent** | Core evaluation logic unchanged. |

---

## Strategy 3: Full Reimplementation (MACS Pattern)

### Description

Reimplement the Tau 2 benchmark from scratch using MASEval's abstractions, referencing tau2-bench for specification and validation. Download and use tau2-bench's data files but implement all evaluation logic natively.

### Implementation Approach

1. Study tau2-bench's domain specifications and evaluation methodology
2. Implement `Tau2GenericTool` similar to `MACSGenericTool` (using existing `ToolLLMSimulator`)
3. Implement `Tau2Environment` loading domains from tau2 data files
4. Implement `Tau2User` extending MASEval's `User` class (using existing `UserLLMSimulator`)
5. Implement `Tau2Evaluator` with Pass^k metric computation
6. Create data loader for tau2 domain files (airline, retail, telecom)
7. Validate results against tau2-bench reference outputs

### Trade-off Analysis

#### 1. Licensing Compatibility

| Rating        | Analysis                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------- |
| **Excellent** | Data files are MIT licensed. Reimplementation creates new code owned by MASEval. No licensing concerns. |

#### 2. Maintenance & Upstream Sync

| Rating       | Analysis                                                                                                                                                                                                                         |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Moderate** | No automatic sync. However, tau2-bench's domain data (the unique asset) is relatively stable. Evaluation methodology changes would require manual porting, but the core logic (Pass^k, multi-turn dialogues) is straightforward. |

**Mitigation**: The truly complex parts (user simulation, tool simulation, multi-turn orchestration) are already implemented in MASEval core. Only domain-specific data loading and metric aggregation need to be maintained.

#### 3. Ease of Use / Installation

| Rating        | Analysis                                                                                                                          |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent** | Zero external dependencies beyond MASEval's core. Data downloaded at runtime (like MACS). Clean `pip install maseval` experience. |

#### 4. Architectural Consistency

| Rating        | Analysis                                                                                                                                                                                                         |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent** | Perfect alignment with MASEval patterns. Uses native `Environment`, `Evaluator`, `User`, `Benchmark` classes. Consistent with MACS implementation. Single codebase style. Full callback and tracing integration. |

#### 5. Reproducibility

| Rating                   | Analysis                                                                                                                                                          |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Good with Validation** | Pass^k is mathematically simple (success rate across k trials). Multi-turn interaction uses the same LLM prompts. Risk of divergence is low and can be validated. |

**Reproducibility Analysis**:

The tau2-bench evaluation consists of:

1. **Pass^k metric**: `success = any(run.passed for run in runs[:k])` — trivial to implement correctly
2. **Task success criteria**: Domain-specific checks (e.g., "flight was booked correctly") — defined in data files
3. **User simulation**: LLM generates user responses — same prompts produce same behavior
4. **Tool simulation**: LLM generates tool responses — same prompts produce same behavior

None of these involve proprietary algorithms or complex logic that would be difficult to replicate accurately.

---

## Strategy 4: Vendoring (Copy Source)

### Description

Copy tau2-bench's source code directly into MASEval's repository under `maseval/benchmark/tau2/_vendor/`. Modify as needed to integrate with MASEval's patterns.

### Trade-off Analysis

#### 1. Licensing Compatibility

| Rating        | Analysis                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------- |
| **Excellent** | MIT allows copying and redistribution. Must preserve copyright notice in vendored files. |

#### 2. Maintenance & Upstream Sync

| Rating        | Analysis                                                                                                                                                                            |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Very Poor** | Highest maintenance burden. Approximately 12+ Python modules with 1000s of lines of code to maintain. Vendored code diverges over time, making future syncs increasingly difficult. |

#### 3. Ease of Use / Installation

| Rating       | Analysis                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------ |
| **Moderate** | No Git install, but tau2-bench's 29 dependencies must still be added to MASEval's optional deps. |

#### 4. Architectural Consistency

| Rating   | Analysis                                                                                                                                       |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Poor** | Creates a foreign codebase within MASEval. Different coding style (88 char lines vs MASEval's 144), different patterns, different conventions. |

#### 5. Reproducibility

| Rating                 | Analysis                                                                                                                               |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Degrades Over Time** | Initially identical, but modifications for integration cause divergence. Harder to validate against upstream as codebases drift apart. |

---

## Comparison Matrix

| Criterion                   | Strategy 1: Git Dep | Strategy 2: Selective Import | Strategy 3: Reimplementation | Strategy 4: Vendoring |
| --------------------------- | ------------------- | ---------------------------- | ---------------------------- | --------------------- |
| **License Compatibility**   | Excellent           | Excellent                    | Excellent                    | Excellent             |
| **Maintenance Burden**      | Low                 | Low-Medium                   | Medium                       | Very High             |
| **Upstream Sync**           | Automatic           | Semi-automatic               | Manual (data only)           | Manual (full code)    |
| **Installation UX**         | Moderate            | Moderate                     | Excellent                    | Moderate              |
| **Arch. Consistency**       | Poor                | Moderate                     | Excellent                    | Poor                  |
| **Reproducibility**         | Excellent           | Excellent                    | Good (validatable)           | Degrades              |
| **Dependency Footprint**    | 29 packages         | 29 packages                  | 0 packages                   | 29 packages           |
| **Development Effort**      | Low                 | Medium                       | Medium                       | Medium-High           |
| **System Benchmarking Fit** | Poor                | Moderate                     | Excellent                    | Poor                  |

---

## Recommended Strategy

### Decision: Strategy 3 — Full Reimplementation (MACS Pattern)

After careful analysis of MASEval's existing capabilities and the actual unique value provided by tau2-bench, **Strategy 3 (Full Reimplementation)** provides the best balance for MASEval's mission.

### Rationale

#### 1. MASEval Already Has the Hard Parts

The complex infrastructure that would justify wrapping tau2-bench is **already implemented in MASEval**:

| Capability                   | MASEval Component                   | Status |
| ---------------------------- | ----------------------------------- | ------ |
| Multi-turn agent-user loops  | `Benchmark.execution_loop()`        | Exists |
| User response simulation     | `User` + `UserLLMSimulator`         | Exists |
| Tool response simulation     | `ToolLLMSimulator`                  | Exists |
| Task repetition (for Pass^k) | `Benchmark.n_task_repeats`          | Exists |
| Early termination            | `User.stop_token`, `is_done()`      | Exists |
| Trace collection             | `TraceableMixin`, `gather_traces()` | Exists |
| Callback system              | `BenchmarkCallback`                 | Exists |

**What remains to implement**:

- Data loader for tau2 domain files (~200 lines)
- Domain-specific tool definitions (~300 lines)
- Pass^k metric aggregation (~50 lines)
- Validation against reference outputs (one-time effort)

#### 2. Zero Dependency Overhead

tau2-bench brings 29 dependencies, most of which MASEval doesn't need:

```
Unused tau2-bench dependencies:
- FastAPI, uvicorn → MASEval doesn't serve HTTP endpoints
- gymnasium → MASEval doesn't do RL training
- pandas, scikit-learn → MASEval uses simple data structures
- plotly, seaborn, matplotlib → MASEval doesn't generate plots
- redis → MASEval uses callbacks for caching/logging
```

Strategy 3 adds **zero dependencies** while Strategies 1, 2, and 4 all inherit this bloat.

#### 3. System Benchmarking Flexibility

MASEval's mission is **system benchmarking**, not model benchmarking:

| Aspect           | tau2-bench Approach | MASEval Needs                                    |
| ---------------- | ------------------- | ------------------------------------------------ |
| What varies      | LLM model           | Agent architecture, prompts, tools, coordination |
| Evaluation focus | Model capability    | System design effectiveness                      |
| Comparison unit  | "GPT-4 vs Claude"   | "ReAct vs CoT vs Multi-agent"                    |

A native implementation allows MASEval to:

- Add system-level metrics (agent communication patterns, tool usage efficiency)
- Integrate with MASEval's callback system for custom tracing
- Extend evaluation criteria beyond tau2-bench's model-centric view

#### 4. Reproducibility is Achievable

The reproducibility concern is overstated because:

1. **Pass^k is mathematically trivial**:

   ```python
   pass_k = any(run.success for run in task_runs[:k])
   ```

2. **User/tool simulation uses the same LLM prompts**: Same inputs → same outputs

3. **Validation is straightforward**: Run both implementations on the same tasks, compare Pass^k scores. Acceptable tolerance: ±2-3% (accounts for LLM stochasticity)

4. **Domain data is the source of truth**: The task specifications, policy constraints, and evaluation criteria come from data files, not code

### Implementation Roadmap

```
Phase 1: Data Infrastructure
├── maseval/benchmark/tau2/data_loader.py
│   ├── download_domain_data()  # Fetch from tau2-bench repo
│   ├── load_domain(name)       # Parse airline/retail/telecom
│   └── load_tasks(domain)      # Convert to TaskCollection

Phase 2: Core Components
├── maseval/benchmark/tau2/tau2.py
│   ├── Tau2GenericTool        # Extends MACSGenericTool pattern
│   ├── Tau2Environment        # Domain-aware tool creation
│   ├── Tau2User               # Multi-turn user with tau2 profiles
│   ├── Tau2Evaluator          # Pass^k and task success metrics
│   └── Tau2Benchmark          # Orchestrates execution

Phase 3: Framework Examples
├── examples/tau2_benchmark/
│   ├── tau2_smolagents.py     # Smolagents implementation
│   └── tau2_langgraph.py      # LangGraph implementation

Phase 4: Validation
├── scripts/validate_tau2.py   # Compare results with upstream
└── tests/test_benchmarks/test_tau2/
    ├── test_tau2_evaluator.py
    ├── test_tau2_environment.py
    └── test_tau2_integration.py
```

### Validation Protocol

To ensure reproducibility, implement a validation script:

```python
def validate_against_upstream():
    """
    1. Run tau2-bench on reference tasks with fixed seed
    2. Run MASEval Tau2Benchmark on same tasks with same seed
    3. Compare Pass^k scores
    4. Assert difference < 3% (accounts for LLM stochasticity)
    """
```

Document any intentional deviations (e.g., MASEval-specific metrics) clearly.

### Why Not Other Strategies?

| Strategy                         | Rejection Reason                                                                                                         |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Strategy 1: Full Git Dep**     | 29 unnecessary dependencies. Poor architectural fit. tau2-bench's orchestration duplicates MASEval's `execution_loop()`. |
| **Strategy 2: Selective Import** | Still pulls all 29 dependencies. Internal API coupling risk. Impedance mismatch with MASEval's patterns.                 |
| **Strategy 4: Vendoring**        | Worst of all worlds: 29 dependencies + highest maintenance burden + divergence over time.                                |

---

## Appendix: License Attribution

When implementing Strategy 3, include the following in `maseval/benchmark/tau2/__init__.py`:

```python
"""
Tau 2 Benchmark Integration for MASEval.

This module implements the tau2-bench benchmark (https://github.com/sierra-research/tau2-bench)
using MASEval's native abstractions. Domain data is downloaded from the original repository.

tau2-bench was developed by Sierra Research and is licensed under the MIT License.
Copyright (c) 2025 Sierra Research
See: https://github.com/sierra-research/tau2-bench/blob/main/LICENSE

This implementation follows the tau2-bench evaluation methodology while adapting it
to MASEval's system-benchmarking focus.
"""
```
