# Getting Started

This guide will help you get started with MASEval.

## Installation

MASEval is designed with a modular architecture:

- **Core**: Framework-agnostic benchmark infrastructure (tasks, evaluation, simulation)
- **Interface**: Optional adapters for specific agent frameworks (smolagents, langgraph, etc.)

Install only what you need to keep dependencies minimal.

### Core Installation

Install the base package from PyPI:

```bash
pip install maseval
```

This includes all core functionality for defining benchmarks, tasks, and evaluators.

### Optional Dependencies

Install additional integrations based on your agent framework and tooling. These can also be installed separately, but are offered here for convenience.

**Agent Frameworks:**

```bash
pip install "maseval[smolagents]"  # SmolAgents integration
```

**LLM Providers:**

```bash
pip install "maseval[openai]"      # OpenAI models
pip install "maseval[google]"      # Google GenAI models
```

**Observability & Tracing:**

```bash
# TODO
```

**Combine Multiple Extras:**

```bash
pip install "maseval[smolagents,openai,wandb]"
```

**Install Everything:**

```bash
pip install "maseval[all]"         # All integrations
pip install "maseval[examples]"    # All dependencies needed for examples
```

## Use the Library

Start with examples.

TODO: Insert examples

## Use the Docs

The docs are hosted here: TODO

For comprehensive documentation on how to piece together the library's components—including detailed explanations of the execution lifecycle, setup methods, and best practices — see the [`Benchmark`](../reference/benchmark.md) class documentation.
