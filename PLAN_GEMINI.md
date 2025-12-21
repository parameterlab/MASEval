# Plan: Porting Tau 2 Benchmark to MASEval (Gemini)

## 1. Executive Summary

**Strategy:** Full Re-implementation (Porting).
**Rationale:** This approach ensures deep integration with MASEval's tracing and callback systems, removes heavy upstream dependencies (Gymnasium), and aligns with the existing `MACS` benchmark architecture. It offers the best user experience ("batteries included") while maintaining the "Core vs. Interface" separation principle.

Important: Existing code should be used as much as possible inline with this strategy. There is no value in reinventing the wheel unless we have to.

## 2. Architectural Blueprint

We will invert the control flow. Instead of the `tau2` harness driving the agent, `MASEval` will drive the agent, and the `Tau2` components will serve as the Environment and Evaluator.

### 2.1. File Structure

We will maintain a modular structure to separate data loading, core logic, and domain specifics, following the user's preference for multi-file organization.

```text
maseval/benchmark/tau2/
├── __init__.py           # Exports public API
├── environment.py        # Tau2Environment (extends maseval.core.Environment)
├── evaluator.py          # Tau2Evaluator (extends maseval.core.Evaluator)
├── data_loader.py        # Logic to download/parse upstream JSONs (from GitHub)
├── metrics.py            # Pass@k computation logic
├── agent.py              # ReferenceAgent (for validation only)
├── PROVENANCE.md         # Strict mapping of files/functions to upstream
├── prompt_templates/     # Extracted system/user prompts
│   ├── user.txt
│   └── evaluator.txt
└── domains/              # Domain-specific logic ported from upstream
    ├── __init__.py
    ├── retail/
    │   ├── tools.py      # Ported tool logic (the "business logic" engine)
    │   └── models.py     # Ported Pydantic data models
    ├── airline/
    │   └── ...
    └── ...
```

### 2.2. Component Specifications

#### A. `Tau2TaskLoader` (in `data_loader.py`)

**Responsibility:** Downloads data from the upstream repo (if missing) and parses it into MASEval `Task` objects.
**Upstream Source:** `data/tau2/domains/{domain}/tasks.json`, `db.json`, `policy.md`.

```python
def download_domain_data(domain: str, data_dir: Path) -> None:
    # Logic to fetch raw JSON/Markdown from sierra-research/tau2-bench via HTTP
    pass

def load_tasks(domain: str, split: str = "test") -> List[Task]:
    # 1. Ensure data exists (call download_domain_data if needed)
    # 2. Parse into upstream Pydantic models
    # 3. Map to maseval.Task(...)
    pass
```

#### B. `Tau2Environment` (in `environment.py`)

**Responsibility:** Manages the state of the simulated world (Retail, Airline, etc.), provides tools to agents, and handles state synchronization.
**Upstream Source:** `src/tau2/environment/environment.py`.

```python
class Tau2Environment(Environment):
    """
    Manages domain-specific state (DBs) and toolsets.
    """
    def __init__(self, task_data: Dict[str, Any], domain: str, ...):
        # Initialize domain-specific DBs using ported Pydantic models
        pass

    def create_tools(self) -> Dict[str, MACSGenericTool]:
        # Convert upstream ToolKit tools into callables wrapped in MACSGenericTool
        pass

    def get_db_hash(self) -> str:
        # Returns hash of current internal state (for reproducibility checks)
        pass
```

#### C. `Tau2Evaluator` (in `evaluator.py`)

**Responsibility:** Implements the composite evaluation logic (DB checks, Action checks, Communication checks).
**Upstream Source:** `src/tau2/evaluator/evaluator.py`.

#### D. `Metrics` (in `metrics.py`)

**Responsibility:** Compute **Pass@k**, the probability that at least one of `k` attempts succeeds. This is the primary metric used in the original Tau 2 paper.

#### E. `Tau2ReferenceAgent` (in `agent.py`)

**Responsibility:** A baseline agent that strictly implements the logic used in the upstream repo.
**Usage:** VALIDATION ONLY. Used to verify that our Environment/Evaluator produces identical scores to the original benchmark.

## 3. Provenance & Attribution Strategy

To respect the MIT license and aid future maintenance:

1.  **PROVENANCE.md:** A dedicated file mapping our files to upstream files.
2.  **Docstrings:** Every file must cite the upstream source and license.
3.  **Prompts:** Extract prompts to `maseval/benchmark/tau2/prompt_templates/` to avoid "hidden" behavior and ensure transparency.

## 4. Implementation Steps

1.  **Scaffolding:** Create directory structure and `PROVENANCE.md`.
2.  **Domain Porting (Retail):** Port `retail` models and tools first to establish the pattern.
3.  **Core Logic:** Implement `Tau2Environment` and `Tau2Evaluator`.
4.  **Data Loading:** Implement `data_loader.py` with GitHub downloader.
5.  **Metrics:** Implement `metrics.py` for Pass@k aggregation.
6.  **Contract Test:**
    - Load a retail task.
    - Run a hardcoded sequence of tool calls known to succeed.
    - Verify the DB hash and Evaluator output match upstream.
7.  **Reference Agent:** Port the upstream agent and run end-to-end validation.

## 5. Risk Assessment & Mitigation

| Risk                     | Impact | Mitigation                                                                                                                  |
| :----------------------- | :----- | :-------------------------------------------------------------------------------------------------------------------------- |
| **Logic Divergence**     | High   | Use a "Contract Test" (Step 6) to verify that a sequence of tool calls produces the _exact_ same DB state hash as upstream. |
| **Dependencies**         | Medium | Match upstream Pydantic validation logic carefully during porting.                                                          |
| **Prompt Inconsistency** | Medium | Centralize all prompts in `prompt_templates/` to ensure we are using the exact same instructions as the original benchmark. |
