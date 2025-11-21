# Testing Strategy for Optional Dependencies

## Overview

Tests are organized into two categories:

- **Core tests**: Run without optional dependencies
- **Interface tests**: Run with all optional dependencies installed

## Running Tests Locally

### Core tests only (minimal environment)

```bash
# In your .venv
pytest -m core -v
```

### All tests (requires optional dependencies)

```bash
# In your .venv with all deps installed
pytest -v
```

### Specific integration tests

```bash
pytest -m smolagents -v
pytest -m interface -v
pytest -m contract -v  # Run cross-implementation contract tests
```

## Test Markers

Defined in `pyproject.toml`:

- `core`: Tests requiring no optional dependencies
- `interface`: Tests requiring optional dependencies
- `contract`: Cross-implementation contract tests that validate framework-agnostic abstraction
- `smolagents`: Tests specifically for smolagents integration
- `langgraph`: Tests specifically for langgraph integration
- `crewai`: Tests specifically for crewai integration

## GitHub Actions Workflow

Two jobs run in sequence:

### 1. `test-core` (Python 3.10, 3.11, 3.12)

- Installs only core package
- Runs `pytest -m core`
- Must pass before interface tests run

### 2. `test-all` (Python 3.10, 3.11, 3.12)

- Depends on `test-core` passing
- Installs all optional dependencies
- Runs all tests including interface tests

## Test Files

```
tests/
├── test_core/
│   └── test_benchmark.py          # Core functionality (marked with @pytest.mark.core)
└── test_interface/
    ├── test_optional_imports.py   # Import behavior without deps (marked core)
    └── test_smolagents_integration.py  # Smolagents integration (marked smolagents)
```

## Implementation Details

### Test Markers Usage

```python
# Mark entire file
pytestmark = pytest.mark.core

# Mark individual test
@pytest.mark.interface
def test_something():
    pass
```

### Skipping Tests Without Dependencies

```python
# At module level - skip entire file if package not available
pytest.importorskip("smolagents")
```

## Test Organization

Tests are organized into three directories following a **bottom-up and top-down strategy**:

### `test_core/`

**Bottom-up unit tests** for core classes in isolation (single implementation). These test individual components without optional dependencies.

Examples:

- `test_model_adapter.py` - Base `ModelAdapter` class behavior
- `test_agent_adapter.py` - Base `AgentAdapter` class behavior
- `test_benchmark_lifecycle.py` - Core benchmark orchestration

### `test_interface/`

**Bottom-up integration tests** for framework-specific adapters (single framework at a time). These test integration with external frameworks like smolagents, langgraph, OpenAI, Google GenAI, etc.

Examples:

- `test_agent_integration/` - Framework-specific agent adapters
- `test_model_integration/` - Provider-specific model adapters (OpenAI, Google, HuggingFace, LiteLLM)

### `test_contract/`

**Top-down contract tests** validate that different implementations of the same abstraction honor identical behavioral contracts. These are the **most critical tests** for MASEval's framework-agnostic promise.

Contract tests use parametrized tests to verify that all implementations (e.g., different framework adapters) behave identically for key operations:

- `test_agent_adapter_contract.py` - All `AgentAdapter` implementations return same message format, trigger callbacks uniformly
- `test_model_adapter_contract.py` - All `ModelAdapter` implementations log calls identically, produce same trace/config structure (65+ parameterized tests)
- `test_collection_contract.py` - All components (Agent, Model, Environment, User) follow same tracing/config contracts

**Why Contract Tests Matter:** MASEval's core value proposition is framework-agnostic abstraction. Users should be able to swap between agent frameworks (smolagents, langgraph, custom) or model providers (OpenAI, Google, HuggingFace) without changing benchmark code. Contract tests validate this promise.

**Test Strategy:** Contract tests create all implementations with mock clients, run identical operations, and assert results match exactly (same fields, types, behavior).
