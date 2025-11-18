# AGENTS.md

## Project Overview

MASEval is an orchestration library for benchmarking LLM-based multi-agent systems. Think of it as PyTorch Lightning for agent evaluation—it provides the execution engine while users implement agent logic.

**Key Architecture Rule:** Strict separation between `maseval/core` (minimal dependencies) and `maseval/interface` (optional integrations). Core must NEVER import from interface.

The library is in early development, so breaking changes that are parsimonous are strongly preferred.

## Setup Commands

```bash
# Sync environment with all dependencies (creates .venv automatically)
uv sync --all-extras --all-groups

# Activate environment
source .venv/bin/activate

# Or use uv run for individual commands (no activation needed)
uv run python examples/amazon_collab.py
uv run pytest tests/
```

## Code Style and Quality

- Line length: 144 characters
- Tool: `ruff`
- All checks must pass in CI before merge

```bash
# Format code
ruff format .

# Lint and auto-fix issues
ruff check . --fix
```

## Testing Instructions

- Tests use pytest markers: `core`, `interface`, `smolagents`, `langgraph`, `contract`
- All tests must pass before PR merge
- Add/update tests for code changes
- Fix type errors and lint issues until suite is green

```bash
# Run all tests
pytest -v

# Core tests only (minimal dependencies)
pytest -m core -v

# Specific integration tests
pytest -m smolagents -v
pytest -m interface -v
```

## Dependency Management

Three types of dependencies:

- **Core** (`[project.dependencies]`): Required, installed with `pip install maseval`. Keep minimal!
- **Optional** (`[project.optional-dependencies]`): Published for end users. Framework integrations like `smolagents`, `langgraph`.
- **Dev Groups** (`[dependency-groups]`): NOT published. Only for contributors. Tools like `pytest`, `ruff`, `mkdocs`.

```bash
# Add core dependency (use sparingly!)
uv add <package-name>

# Add optional dependency for end users
uv add --optional <extra-name> <package-name>

# Add development dependency (not published)
uv add --group dev <package-name>

# Remove any dependency
uv remove <package-name>
```

**Important:** `uv add` automatically updates both `pyproject.toml` and `uv.lock`. Never edit lockfile manually.

## Architecture Rules

**Core vs Interface Separation (CRITICAL)**

- `maseval/core/`: Essential library logic. NO optional dependencies. Must work with minimal installation.
- `maseval/interface/`: Adapters for external frameworks. ALL dependencies are optional.

**NEVER:**

- Import `maseval/interface` from `maseval/core`
- Add optional dependencies to core
- This is enforced in CI and will fail the build

**Adding new framework integrations:**

1. Create adapter in `maseval/interface/<library>/`
2. Add dependency to `[project.optional-dependencies]`
3. Add tests in `tests/test_interface/`
4. Mark tests with appropriate pytest marker
5. Keep adapters small and well-documented

**Framework Adapter Pattern:**

When implementing wrappers for external frameworks, **always use the framework's native message storage as the source of truth**:

**Pattern 1: Persistent State (smolagents)**

```python
class MyFrameworkWrapper(AgentAdapter):
    def get_messages(self) -> MessageHistory:
        """Dynamically fetch from framework's internal storage."""
        # Get from framework (e.g., agent.memory, agent.messages)
        framework_messages = self.agent.get_messages()

        # Convert and return immediately (no caching)
        return self._convert_messages(framework_messages)

    def _run_agent(self, query: str) -> MessageHistory:
        # Run agent (updates framework's internal state)
        self.agent.run(query)

        # Return fresh history
        return self.get_messages()
```

**Pattern 2: Stateless/Result-based (LangGraph)**

```python
def __init__(self, agent_instance, name, callbacks=None, config=None):
    super().__init__(agent_instance, name, callbacks)
    self._last_result = None  # Cache for stateless mode
    self._config = config  # For stateful mode if supported

def get_messages(self) -> MessageHistory:
    # Try persistent state first (if framework supports it)
    if self._config and hasattr(self.agent, 'get_state'):
        state = self.agent.get_state(self._config)
        return self._convert_messages(state.values['messages'])

    # Fall back to cached result
    if self._last_result:
        return self._convert_messages(self._last_result['messages'])

    return MessageHistory()

def _run_agent(self, query: str) -> MessageHistory:
    result = self.agent.invoke(query, config=self._config)
    self._last_result = result  # Cache result
    return self.get_messages()
```

**Why no caching?** Conversion is cheap, and caching creates consistency issues if framework state changes. The framework's internal storage is the single source of truth. For stateless frameworks, cache only the last result and prefer fetching from persistent state if available.

## Documentation

- Built with MkDocs Material theme + mkdocstrings
- Source files in `docs/`, config in `mkdocs.yml`

```bash
# Build with strict checking (catches broken links)
mkdocs build --strict

# Serve locally at http://127.0.0.1:8000
mkdocs serve
```

## Contribution Workflow

1. Create a feature branch (never commit to `main`)
2. Make changes following code style guidelines
3. Run formatters and linters: `ruff format . && ruff check . --fix`
4. Run tests: `pytest -v`
5. Update documentation if needed
6. Open PR against `main` branch
7. Request review from `cemde`
8. Ensure all CI checks pass

**CI Pipeline:** GitHub Actions runs formatting checks, linting, and test suite across Python versions and OS. All checks must pass before merge.

## For VS Code Agents Specifically

**Important:** VS Code agents (GitHub Copilot in VS Code) do NOT have direct access to run terminal commands like `uv`, `python`, `pytest`, or `ruff`.

When you need to run commands:

- Ask the user to execute commands
- Provide clear explanations of what each command does
- Don't attempt to run `uv`, `python`, `pytest`, or similar commands directly, it does not work.
- You CAN read/write files, and execute simple terminal commands such as `ls`, `cd`, `grep` etc.

Example workflow:

1. Make code changes using file edit tools
2. Ask user to run: `ruff format . && ruff check . --fix`
3. Ask user to run: `pytest -v` to verify changes
4. Read test output if user provides it

## Project-Specific Conventions

- Minimum Python: 3.10 (Primary development: 3.12)
- Package manager: `uv` exclusively (don't use `pip install` in development)
- Commit guidelines: Reference issues, keep commits focused, run full test suite before pushing

## Common Tasks Quick Reference

```bash
# Fresh environment setup
uv sync --all-extras --all-groups

# Before committing
ruff format . && ruff check . --fix && pytest -v

# Run example
uv run python examples/amazon_collab.py

# Update after pulling changes
uv sync --all-extras --all-groups

# Add optional dependency
uv add --optional <extra-name> <package-name>

# Check specific test file
pytest tests/test_core/test_agent.py -v
```

## Type Hinting

This repository uses proper type hinting. For unions use the `A | B` notation. For optional imports, prefer `Optional[...]` as it is more explicit.
For lists and dictionaries, use `Dict[...,...]`, `List[...]`, `Sequence[...]` etc. instead of `list`, `dict`.

## Security and Confidentiality

**IMPORTANT:** This project contains confidential research material.

- DO NOT publicly distribute code or data
- DO NOT publish without explicit permission
- DO NOT share copyrighted third-party benchmark data
