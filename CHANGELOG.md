# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Benchmarks**

- MACS Benchmark: Multi-Agent Collaboration Scenarios benchmark (PR: #13)

**Benchmark**

- Added `execution_loop()` method to `Benchmark` base class enabling iterative agent-user interaction (PR: #13)
- Added `max_invocations` constructor parameter to `Benchmark` (default: 1 for backwards compatibility) (PR: #13)
- Added abstract `get_model_adapter(model_id, **kwargs)` method to `Benchmark` base class as universal model factory to be used throughout the benchmarks. (PR: #13)

**User**

- Added `max_turns` and `stop_token` parameters to `User` base class for multi-turn support with early stopping (PR: #13)
- Added `is_done()`, `_check_stop_token()`, and `increment_turn()` methods to `User` base class (PR: #13)
- Added `get_initial_query()` method to `User` base class for LLM-generated initial messages (PR: #13)
- Added `initial_prompt` parameter in `User` base class to trigger the agentic system. (PR: #13)

**Environment**

- Added `Environment.get_tool(name)` method for single-tool lookup (PR: #13)

**Interface**

- [LlamaIndex](https://github.com/run-llama/llama_index) integration: `LlamaIndexAgentAdapter` and `LlamaIndexUser` for evaluating LlamaIndex workflow-based agents (PR: #7)
- The `logs` property inside `SmolAgentAdapter` and `LanggraphAgentAdapter` are now properly filled. (PR: #3)

**Examples**

- Added a new example: The `5_a_day_benchmark` (PR: #10)

### Changed

**Environment**

- `Environment.create_tools()` now returns `Dict[str, Any]` instead of `list` (PR: #13)

**Benchmark**

- `Benchmark.run_agents()` signature changed: added `query: str` parameter (PR: #13)
- `Benchmark.run()` now uses `execution_loop()` internally to handle agent-user interaction cycles (PR: #13)
- `Benchmark` class now has a `fail_on_setup_error` flag that raises errors observed during setup of task (PR: #10)

**Callback**

- `FileResultLogger` now accepts `pathlib.Path` for argument `output_dir` and has an `overwrite` argument to prevent overwriting of existing logs files.

**Evaluator**

- The `Evaluator` class now has a `filter_traces` base method to conveniently adapt the same evaluator to different entities in the traces (PR: #10).

**Other**

- Documentation formatting improved. Added darkmode and links to `Github` (PR: #11).
- Improved Quick Start Guide in `docs/getting-started/quickstart.md`. (PR: #10)
- `maseval.interface.agents` structure changed. Tools requiring framework imports (beyond just typing) now in `<framework>_optional.py` and imported dynamically from `<framework>.py`. (PR: #12)
- Various formatting improvements in the documentation (PR: #12)
- Added documentation for View Source Code pattern in `CONTRIBUTING.md` and `_optional.py` pattern in interface README (PR: #12)

### Fixed

**Interface**

- `LlamaIndexAgentAdapter` now supports multiple LlamaIndex agent types including `ReActAgent` (workflow-based), `FunctionAgent`, and legacy agents by checking for `.chat()`, `.query()`, and `.run()` methods in priority order (PR: #10)

**Other**

- Consistent naming of agent `adapter` over `wrapper` (PR: #3)
- Fixed an issue that `LiteLLM` interface and `Mixin`s were not shown in documentation properly (#PR: 12)

### Removed

- Removed `set_message_history`, `append_message_history` and `clear_message_history` for `AgentAdapter` and subclasses. (PR: #3)

## [0.1.2] - 2025-11-18

### Added

- Automated release workflow with version verification
- Documentation for release process

### Changed

- Improved project documentation structure

## [0.1.1] - [Previous release date]

<!-- Previous changes here -->

[Unreleased]: https://github.com/parameterlab/maseval/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/parameterlab/maseval/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/parameterlab/maseval/releases/tag/v0.1.1
