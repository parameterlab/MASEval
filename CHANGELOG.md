# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [LlamaIndex](https://github.com/run-llama/llama_index) integration: `LlamaIndexAgentAdapter` and `LlamaIndexUser` for evaluating LlamaIndex workflow-based agents (PR: #7)
  - Supports async workflow execution with proper event loop handling
- Added a new example: The `5_a_day_benchmark` (PR: #10)
- The `logs` property inside `SmolAgentAdapter` and `LanggraphAgentAdapter` are now properly filled. (PR: #3)

### Changed

- Documentation formatting improved. Added darkmode and links to `Github` (PR: #11).
- `FileResultLogger` now accepts `pathlib.Path` for argument `output_dir` and has an `overwrite` argument to prevent overwriting of existing logs files.
- `Benchmark` class now has a `fail_on_setup_error` flag that raises errors observed during setup of task (PR: #10)
- The `Evaluator` class now has a `filter_traces` base method to conveniently adapt the same evaluator to different entities in the traces (PR: #10).
- Improved Quick Start Guide in `docs/getting-started/quickstart.md`. (PR: #10)
- `maseval.interface.agents` structure changed. Tools requiring framework imports (beyond just typing) now in `<framework>_optional.py` and imported dynamically from `<framework>.py`.

### Fixed

- `LlamaIndexAgentAdapter` now supports multiple LlamaIndex agent types including `ReActAgent` (workflow-based), `FunctionAgent`, and legacy agents by checking for `.chat()`, `.query()`, and `.run()` methods in priority order (PR: #10)
- Consistent naming of agent `adapter` over `wrapper` (PR: #3)

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
