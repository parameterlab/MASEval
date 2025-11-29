# Five-A-Day Benchmark - Agent Instructions

## General Agent Instructions

You are modifying a **static, self-contained example** that exists only to showcase the flexibility of the `maseval` library.

This code is **pre-release** and **intentionally simple**. Educational clarity is more important than robustness or feature completeness.

## CRITICAL: CHANGELOG Context Awareness

**DO NOT update the main repository CHANGELOG.md with implementation details from work within this example.**

The main `/CHANGELOG.md` is for tracking changes **between PRs and releases**, not internal implementation details within a single PR or example. When working on this benchmark:

- ❌ DO NOT add entries like "Refactored tools in Five-A-Day" or "Fixed bug in Five-A-Day tool X"
- ❌ DO NOT document internal iterations, bug fixes, or refactorings done during development
- ✅ DO update the main CHANGELOG only when this entire example is being added/modified in a PR that affects the published library
- ✅ DO document significant changes in this local `examples/5_a_day_benchmark/AGENTS.md` or README if needed

**Why?** If a feature is being introduced in PR #10, all the work to implement it correctly is part of that PR's scope. The CHANGELOG should say "Added Five-A-Day benchmark (PR #10)", not list every internal implementation detail. Think of the bigger picture: what matters to users consuming releases, not what happened during development.

---

## Goals

1. **Showcase the library’s flexibility** with a small, readable example.
2. **Prioritize simplicity and clarity** over cleverness, performance, or robustness.
3. **Make it easy for a human reader to understand** what the code is doing and how it uses `maseval`.

---

## Non-Goals

- Do **NOT** optimize for performance or handle all edge cases.
- Do **NOT** build a production-ready tool.
- Do **NOT** add features that are unrelated to demonstrating `maseval`.

---

## Hard Rules

When editing or writing code for this example, you MUST follow these rules:

### 1. Simplicity First

- Prefer straightforward, explicit code over abstractions, helpers, or “clever” patterns.
- Small, linear scripts are preferred over complex architectures.

### 2. Breaking Changes Are Allowed

- This is pre-release example code.
- You **MAY introduce breaking changes** if they **clearly increase simplicity or clarity.**

### 3. Config & JSON Files

- If you request a config from one of the JSON files, **assume that it exists.**
- If it does **not** exist:
  - You MAY **add** the minimal config needed for the example to work.
  - Keep any new config fields simple and clearly named.
- Do **NOT** add complex or unused configuration options.

### 4. Error Handling & Edge Cases

- Do **NOT** try to catch or handle every edge case.
- If something is misconfigured or missing, it is acceptable for the script to raise an error and fail.
- Only add error handling if it **directly helps readability** (e.g., a clear `ValueError` with an explanatory message).

### 5. Defaults & Arguments

- Apart from `argparse` and optional function arguments (e.g. `limit = None`), **do NOT use defaults.**
- If something isn’t set properly (e.g., required argument, required config), the script **should fail**.
- Avoid hidden defaults in functions, classes, or configuration. Make assumptions explicit in the code or CLI. This helps simplicity.

### 6. Argparse Usage

- You MAY use `argparse` for CLI argument parsing.
- Arguments should have defaults as needed for basic usability.
- Required behavior or configuration SHOULD be explicit in arguments or config, not silently defaulted.

### 7. Library Principles

- Follow the principles of the `maseval` library.

### 8. Scope of Changes

- Limit changes to the files that are part of this example unless explicitly instructed otherwise.
- Do **NOT** refactor or redesign unrelated parts of the project.

### 9. Duplicate Code

- Avoid duplicate code.
- Do not introduce new duplication when extending or modifying the example.

---

## Documentation & Comments

The primary purpose of documentation in this example is to **help the user understand the code and learn from it.**

### Docstrings

- Docstrings **MUST explain behavior, inputs, outputs, and important assumptions** in a way that is useful to a user reading the code.
- Focus on _what the function does_ and _how to use it_, not on the history of changes.
