# Tau 2 Benchmark Integration: Implementation Options

This document outlines and analyzes technical strategies for integrating the [Tau 2 Benchmark](https://github.com/sierra-research/tau2-bench) into the MASEval library.

**Reference Repo:** https://github.com/sierra-research/tau2-bench
**License:** MIT (Compatible with MASEval)

## Strategy 1: Full Re-implementation (Porting)

This strategy involves rewriting the benchmark logic natively using MASEval primitives (`Benchmark`, `Task`, `Environment`, `Evaluator`), similar to the existing `MACS` implementation. We would treat the upstream repo primarily as a reference specification and data source.

### Analysis
1.  **Licensing Compatibility:**
    -   **Result:** ✅ Compatible.
    -   Both are MIT. We can adapt the logic and attribute the original authors in the docstrings and module headers.

2.  **Maintenance & Upstream Sync:**
    -   **Result:** ⚠️ High Effort.
    -   If the upstream logic changes (e.g., how the reward is calculated or how the environment transitions), we must manually identify and port those changes.
    -   However, benchmarks are typically static once published.

3.  **Ease of Use / Installation:**
    -   **Result:** 🟢 Excellent.
    -   Zero extra steps for the user. It works immediately with `pip install maseval`.
    -   No need to install heavy external dependencies like `gymnasium` if we can implement the logic with lighter alternatives (or use `maseval`'s existing structures).

4.  **Architectural Consistency:**
    -   **Result:** 🟢 Perfect Alignment.
    -   The implementation will inherit from `maseval.core.Benchmark` and use standard `AgentAdapter` interfaces.
    -   It ensures the strict separation of Core vs. Interface is respected.
    -   It allows us to inject MASEval's tracing and callbacks deeply into the evaluation loop, which might be hard if wrapping a black-box `gymnasium` environment.

5.  **Reproducibility:**
    -   **Result:** ⚠️ Risk Factor.
    -   Rewriting logic introduces the risk of subtle deviations. We must create a "contract test" that runs the upstream code against our implementation to verify they produce identical scores for identical inputs.

## Strategy 2: Source Vendoring (Copying Code)

This strategy involves copying the relevant parts of the `tau2` source code directly into a subdirectory (e.g., `maseval/benchmark/tau2/upstream/`) and modifying it minimally to work within our package structure.

### Analysis
1.  **Licensing Compatibility:**
    -   **Result:** ✅ Compatible.
    -   Permitted under MIT, provided we keep the original license/copyright notices.

2.  **Maintenance & Upstream Sync:**
    -   **Result:** 🟠 Moderate.
    -   Easier to update than a full rewrite (we can copy-paste files again), but local modifications to make it fit MASEval (imports, relative paths) will need to be re-applied every time.

3.  **Ease of Use / Installation:**
    -   **Result:** 🟢 Good.
    -   Users get the code automatically.
    -   **Downside:** We inherit all upstream dependencies. `tau2` requires `gymnasium` and `litellm`. We would need to add these to `maseval`'s dependency tree (likely as an optional extra `maseval[tau2]`).

4.  **Architectural Consistency:**
    -   **Result:** 🔴 Poor.
    -   The upstream code likely uses its own loop, logging, and configuration patterns.
    -   We would need to build a "Wrapper" class that bridges `MASEval` primitives to the vendored internal API. This often feels "bolted on" and may not expose granular tracing hooks.

5.  **Reproducibility:**
    -   **Result:** 🟢 High.
    -   Since we are running the actual original code, results should match exactly.

## Strategy 3: External Dependency (User Managed)

We do not include the code. We require the user to install the benchmark manually (e.g., `pip install git+https://github.com/sierra-research/tau2-bench.git`). We provide a thin adapter class that imports `tau2` at runtime.

### Analysis
1.  **Licensing Compatibility:**
    -   **Result:** ✅ N/A.
    -   We don't distribute their code.

2.  **Maintenance & Upstream Sync:**
    -   **Result:** 🟢 Low Effort.
    -   We don't maintain their code. If they break their API, our adapter breaks, but we don't own the bugs.

3.  **Ease of Use / Installation:**
    -   **Result:** 🔴 Poor.
    -   Since `tau2` is **not on PyPI**, users cannot just do `pip install maseval[tau2]`.
    -   They must run a manual `pip install git+...` command. This is error-prone and friction-heavy for a "batteries included" evaluation library.
    -   Dependency conflicts between `maseval` and `tau2` versions of shared libs (like `pydantic` or `numpy`) become the user's problem to solve.

4.  **Architectural Consistency:**
    -   **Result:** 🟠 Mixed.
    -   Similar to Vendoring, we are wrapping a black box.

5.  **Reproducibility:**
    -   **Result:** ⚠️ Variable.
    -   We cannot control which version of `tau2` the user installs (HEAD vs a specific commit). Results may vary across users if upstream changes.

---

## Recommended Strategy: Strategy 1 (Full Re-implementation)

**Decision:** We should **Port/Re-implement** the Tau 2 benchmark natively into MASEval.

**Justification:**
1.  **UX First:** MASEval aims to be a cohesive evaluation engine. Requiring users to manually `git install` dependencies (Strategy 3) breaks the "one-line install" promise.
2.  **Architectural Integrity:** MASEval's value proposition is its unified tracing and callback system. Wrapping a black-box `gymnasium` environment (Strategy 2/3) often hides the internal "thought process" of the agents or the granular tool outputs, limiting the depth of analysis. Native implementation allows full visibility.
3.  **Precedent:** The `MACS` benchmark is already implemented this way, establishing a clear pattern for the codebase.
4.  **Dependency Control:** We can minimize dependencies. If `tau2` uses `gymnasium` only for the API structure, we can drop that heavy dependency and use our own `Environment` class, keeping the core library lighter.

**Implementation Plan:**
1.  Analyze `tau2` data loading and environment logic.
2.  Create `maseval/benchmark/tau2/` package.
3.  Implement `Tau2Environment` inheriting from `maseval.core.Environment`.
4.  Implement `Tau2Evaluator` for scoring.
5.  Create a strict "Contract Test" that compares outputs of our implementation against the upstream implementation for a fixed seed to ensure 100% reproducibility.
