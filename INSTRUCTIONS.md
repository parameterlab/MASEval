# MultiAgentBench Integration Instructions

## Phase 0: Before You Start

### 0.1 Repeat Back Requirements

List exactly:

- The classes you will design
- The integration approaches you will evaluate
- The design principles you must follow

### 0.2 Create a Scoped ToDo List

List every file/directory you need to examine. Scope constraint: explore MARBLE only as deep as needed for MASEval integration—no deeper. Go into depth on integration-relevant code only.

### 0.3 Periodically Reread

These instructions are saved in `INSTRUCTIONS.md`. Reread and re-summarize requirements:

- Before starting analysis
- After completing each major section
- Before writing final deliverable

---

## Resources

| Resource                  | Location                                      |
| ------------------------- | --------------------------------------------- |
| MARBLE repo               | https://github.com/ulab-uiuc/MARBLE           |
| Local copy                | `/Users/cornelius/Repositories/MARBLE`        |
| Paper                     | `/Users/cornelius/Repositories/MARBLE/paper/` |
| These instructions        | `INSTRUCTIONS.md`                             |
| Reference implementations | `macs` and `tau2` benchmarks in MASEval       |
| MASEval guidelines        | `AGENTS.md`                                   |

**License**: MIT (verify subdirectories)

---

## Deliverable

Write `STRATEGY.md` proposing 2–3 integration approaches with tradeoffs and a recommendation.

Requirements:

- [ ] Standalone document (readable without this conversation)
- [ ] **Do not implement**—strategy document only

---

## Required Analysis: Hybrid Vendoring Approach

Deeply evaluate this specific approach:

**Vendoring**: Clone MARBLE to `maseval/benchmark/multiagentbench/marble/` (gitignored)

**MASEval wrappers** (in `maseval/benchmark/multiagentbench/`):

- [ ] `MultiAgentBenchEnvironment(Environment)`: wraps the environment of MultiAgentBench
- [ ] `MultiAgentBenchEvaluator(Evaluator)`: wraps the evaluation of MultiAgentBnech
- [ ] `MultiAgentBenchBenchmark(Benchmark)` Harness to coordinate environment, eval etc. Does NOT implement specific agents.
- [ ] `MarbleAgentAdapter(AgentAdapter)` Implements the multi-agent engine of MARBLE
- [ ] `MarbleMultiAgentBenchBenchmark(MultiAgentBenchBenchmark)` adapts the benchmark to use the MARBLE agent and runs them with the same parameters as the original to reproduce.
- [ ] Utilities: `load_tasks()`, `load_agent_data()`, `configure_model_ids()`, `ensure_data()`

**Key question**: Do patterns from `tau2` and `macs` transfer smoothly to MultiAgentBench?

---

## Design Principles (Must Address Each)

- [ ] **R1: Reuse MASEval** — Don't reimplement existing features
- [ ] **R2: Scientific fidelity** — Reproduce original results exactly; preserve semantically relevant details; allow functionally equivalent implementation changes
- [ ] **R3: Fail loudly** — No defensive defaults that silently corrupt results; this targets a fixed dataset
- [ ] **R4: Maintainability** — Plan for long-term upkeep

---

## Phase 1: Study

- [ ] Read and internalize `AGENTS.md`
- [ ] Study `macs` benchmark implementation
- [ ] Study `tau2` benchmark implementation
- [ ] Understand MASEval core library (especially `Benchmark` class)
- [ ] Read MARBLE paper (sufficient for integration understanding)
- [ ] Examine MARBLE code (scoped to integration needs)

---

## Phase 2: Analyze

- [ ] Identify 2–3 integration approaches
- [ ] For each approach, document tradeoffs
- [ ] Assess hybrid vendoring approach in depth
- [ ] Determine if `tau2`/`macs` patterns transfer to MultiAgentBench
- [ ] Verify each design principle (R1–R4) is addressed

---

## Phase 3: Write

- [ ] Draft `STRATEGY.md`
- [ ] **Stop and verify**: Reread `INSTRUCTIONS.md`, confirm all checkboxes above are addressed
- [ ] Finalize document

---

## Process Notes

- Question everything
- Do not make assumptions—verify in the codebase
- Take time to consider all consequences
