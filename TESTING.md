# Testing Plan for Tau2 Benchmark & AgenticUser

## Overview of Changes

This branch introduces the **Tau2 Benchmark**, a complex, multi-turn, tool-use benchmark for evaluating agents in customer service domains (Retail, Airline, Telecom). It also adds a core `AgenticUser` component to simulate users that can use tools.

### Key Components Added:
-   `maseval/benchmark/tau2/`: Full benchmark implementation (Environment, Evaluator, User, DataLoader, Domains).
-   `maseval/core/agentic_user.py`: `AgenticUser` and `AgenticUserLLMSimulator` for tool-using user simulation.
-   `examples/tau2_benchmark/`: Example usage and integration scripts.

## Existing Tests

The following test files have been added and cover significant portions of the new functionality:

1.  **`tests/test_core/test_agentic_user.py`**
    -   Covers `AgenticUser` simulation loop.
    -   Verifies tool execution and internal step limits (`max_internal_steps`).
    -   Mocks `ModelAdapter` to ensure correct parsing of JSON output with `tool_calls`.

2.  **`tests/test_benchmarks/test_tau2/test_environment.py`**
    -   Verifies creation of `Tau2Environment` for all domains (Retail, Airline, Telecom).
    -   Checks tool availability and callability.
    -   Verifies Database state hashing (critical for deterministic evaluation).
    -   Checks trace gathering structure.

3.  **`tests/test_benchmarks/test_tau2/test_data_loader.py`**
    -   Verifies `load_tasks` for different splits and domains.
    -   Checks `configure_model_ids` functionality.
    -   Ensures data existence and directory structure validation.

4.  **`tests/test_benchmarks/test_tau2/test_domains/`**
    -   Contains domain-specific tool tests (e.g., `test_retail_tools.py`, `test_telecom_user_tools.py`).

## Recommended Testing Strategy

To ensure robustness and maintainability, the following additional tests are recommended.

### 1. Evaluator Tests (Critical)

**File:** `tests/test_benchmarks/test_tau2/test_evaluator.py`

The `Tau2Evaluator` is complex, handling multiple reward signals (DB state, Actions, Communication). It needs dedicated unit tests.

*   **`test_filter_traces`**: Ensure it correctly extracts messages, tool calls, and environment traces from the raw trace dict.
*   **`test_evaluate_environment`**:
    *   Mock `Tau2Environment` and verify that `db_match` is correctly calculated based on hash comparison.
    *   Verify `env_assertions` logic (running checking functions).
*   **`test_evaluate_actions`**:
    *   Verify correct matching of expected tool calls vs. actual tool calls.
    *   Test argument comparison logic.
*   **`test_evaluate_communication`**:
    *   Verify that required information strings are correctly found in the message history.
*   **`test_score_aggregation`**: Ensure the final `reward` is correctly calculated based on the `reward_basis`.

### 2. Tau2 User Tests (Recommended)

**File:** `tests/test_benchmarks/test_tau2/test_user.py`

While `AgenticUser` is tested in core, `Tau2User` has specific logic for profile extraction and setup.

*   **`test_extract_user_profile`**: Verify that "Persona:" sections are correctly parsed from the scenario string.
*   **`test_get_tool_raises`**: Confirm that the base `Tau2User.get_tool()` raises `NotImplementedError` (as it must be implemented by frameworks).
*   **`test_init`**: Verify that `tools` are correctly passed to the super constructor.

### 3. Benchmark Configuration Tests (Recommended)

**File:** `tests/test_benchmarks/test_tau2/test_benchmark.py`

Test the `Tau2Benchmark` class logic.

*   **`test_setup_user`**: Verify that the user is created with the correct model ID and scenario from the task.
*   **`test_setup_environment`**: Verify environment creation with correct task data.
*   **`test_get_user_model_id_error`**: Ensure a clear `ValueError` is raised if `configure_model_ids` hasn't been called (missing `model_id` in task data).

### 4. Integration / Smoke Test (Optional but Valuable)

**File:** `tests/test_benchmarks/test_tau2/test_integration.py`

A "dry run" to ensure all components wire together correctly without hitting real LLMs.

*   **`test_tau2_dry_run`**:
    *   Subclass `Tau2Benchmark` with a dummy agent.
    *   Use `MockModelAdapter` for the user simulator.
    *   Run a single task.
    *   Assert that `evaluate` is called and returns a result structure.

## Plan of Action

1.  **Implement `test_evaluator.py`**: This is the highest priority as evaluation logic is complex and critical for benchmark validity.
2.  **Implement `test_user.py`**: Low effort, ensures specific Tau2 user logic is correct.
3.  **Implement `test_benchmark.py`**: Ensures configuration safety.
4.  **Run all tests**: `uv run pytest tests/test_benchmarks/test_tau2/` to ensure everything is green.
