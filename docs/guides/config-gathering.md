# Configuration Gathering System

This document describes the new configuration gathering system that mirrors the existing tracing system in MASEval.

## Overview

Similar to how the library gathers execution traces from components during benchmark runs, we now have a parallel system for gathering configuration information. This enables better reproducibility and analysis by capturing all configuration parameters used during benchmark execution.

## Key Components

### 1. ConfigurableMixin (`maseval/core/config.py`)

A new mixin class that provides configuration gathering capability to any component. Similar to `TraceableMixin`, it provides:

- A base `gather_config()` method that returns basic metadata (component type, timestamp)
- Can be extended by subclasses to include component-specific configuration
- Returns JSON-serializable dictionaries

Example usage:

```python
class MyCustomTool(ConfigurableMixin):
    def __init__(self, temperature: float = 0.7, max_retries: int = 3):
        self.temperature = temperature
        self.max_retries = max_retries

    def gather_config(self) -> Dict[str, Any]:
        return {
            **super().gather_config(),
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "version": "1.0.0"
        }
```

### 2. Benchmark Integration

The `Benchmark` class now includes:

#### New Attributes

- `_config_registry`: Registry for configurable components (cleared after each task repetition)
- `_config_component_id_map`: Maps component IDs to registry keys
- `benchmark_configs`: Persistent list storing configs across all task repetitions

#### Updated Methods

**`register(category, name, component)`** - Enhanced to handle both traces and configs

- Now automatically registers components for BOTH trace and configuration collection
- If a component inherits from `ConfigurableMixin`, it's registered for config collection too
- Single method call handles all registration needs
- No need for separate registration methods

**`collect_all_configs()`** - New method for configuration collection

- Collects configuration from all registered components
- Called automatically after each task repetition
- Returns structured dictionary with categories:
  - `metadata`: Collection timestamp and thread info
  - `agents`: Dict mapping agent names to their config
  - `models`: Dict mapping model names to their config
  - `tools`: Dict mapping tool names to their config
  - `simulators`: Dict mapping simulator names to their config
  - `callbacks`: Dict mapping callback names to their config
  - `environment`: Direct config from environment (not nested)
  - `user`: Direct config from user simulator (not nested)
  - `other`: Dict for any other registered components

#### Updated Run Loop

The benchmark run loop now:

1. Collects execution traces (existing)
2. **Collects execution configs (new)**
3. Stores both in persistent lists
4. **Adds config to each result dictionary**

Result structure:

```python
result = {
    "task_id": str(task.id),
    "output": agents_output,
    "eval": eval_results,
    "config": execution_configs  # NEW
}
```

### 3. Core Component Updates

All core components now inherit from both `TraceableMixin` and `ConfigurableMixin`:

- **AgentAdapter**: Includes agent name, type, and callbacks config
- **ModelAdapter**: Includes model_id config (subclasses can add parameters)
- **Environment**: Includes tool count and tool configs
- **User**: Basic config (subclasses can extend)
- **LLMSimulator**: Includes model_id, max_try, and template presence
- **BenchmarkCallback**: Basic config (subclasses can extend)
- **EnvironmentCallback**: Basic config (subclasses can extend)
- **AgentCallback**: Basic config (subclasses can extend)
- **UserCallback**: Basic config (subclasses can extend)

Each component implements `gather_config()` to return relevant configuration information.

### 4. Automatic Registration

Components are automatically registered for both tracing AND configuration when:

- Returned from `setup_environment()`
- Returned from `setup_user()`
- Returned from `setup_agents()`

The `register()` method now automatically calls `register_config()` if the component implements `ConfigurableMixin`.

## Usage

### For Benchmark Implementers

No changes needed! Configuration gathering happens automatically:

```python
class MyBenchmark(Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        model = MyModelAdapter(...)
        agent = MyAgent(model=model)
        wrapper = AgentAdapter(agent, "agent")
        return [wrapper], {"agent": wrapper}
    # ... other methods

# Run benchmark
results = benchmark.run()

# Access configs from results
for result in results:
    print(f"Task {result['task_id']}")
    print(f"Config: {result['config']}")

# Access all configs across repetitions
for config_entry in benchmark.benchmark_configs:
    print(f"Task {config_entry['task_id']}, Repeat {config_entry['repeat_idx']}")
    print(f"Agent config: {config_entry['config']['agents']}")
    print(f"Model config: {config_entry['config']['models']}")
```

### For Component Developers

Extend `gather_config()` in your custom components:

```python
class MyModelAdapter(ModelAdapter, ConfigurableMixin):
    def __init__(self, model_id: str, temperature: float = 0.7, max_tokens: int = 1000):
        super().__init__()
        self._model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self._model_id

    def gather_config(self) -> dict[str, Any]:
        return {
            **super().gather_config(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": "custom"
        }
```

## Design Principles

The configuration system follows the same design principles as the tracing system:

1. **Automatic**: Components are automatically registered and configs collected
2. **Extensible**: Easy to add custom configuration data via `gather_config()`
3. **Non-intrusive**: No changes needed to existing benchmark implementations
4. **Parallel to Tracing**: Same structure and patterns as the tracing system
5. **JSON-serializable**: All configs must be serializable for storage/analysis

## Comparison with Tracing System

| Aspect            | Tracing System                           | Configuration System                                      |
| ----------------- | ---------------------------------------- | --------------------------------------------------------- |
| Mixin Class       | `TraceableMixin`                         | `ConfigurableMixin`                                       |
| Method            | `gather_traces()`                        | `gather_config()`                                         |
| Registry          | `_trace_registry`                        | `_config_registry`                                        |
| Collection Method | `collect_all_traces()`                   | `collect_all_configs()`                                   |
| Storage           | `benchmark_traces`                       | `benchmark_configs`                                       |
| Purpose           | Execution data (messages, calls, errors) | Configuration data (parameters, settings)                 |
| In Results        | Stored in `benchmark_traces` list        | Stored in `benchmark_configs` list and `result['config']` |

## Benefits

1. **Reproducibility**: Full configuration capture enables exact reproduction of benchmark runs
2. **Analysis**: Compare configurations across different runs or tasks
3. **Debugging**: Understand what configuration was used when issues occur
4. **Auditing**: Track configuration changes over time
5. **Documentation**: Automatic documentation of system setup
