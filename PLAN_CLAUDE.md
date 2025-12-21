# Tau 2 Benchmark Implementation Plan

This document provides a detailed technical plan for porting the **Tau 2 benchmark** into MASEval, following the re-implementation strategy recommended in `IMPLEMENTATION_OPTIONS_GEMINI.md` and `IMPLEMENTATION_OPTIONS_CLAUDE.md`.

---

## 1. Strategy Confirmation

### Chosen Approach: Full Re-implementation (Porting)

Both analysis documents converge on **Strategy 3: Full Re-implementation** as the optimal path. This approach:

1. **Aligns with MASEval's architecture** — Uses native `Benchmark`, `Environment`, `Evaluator`, `User` abstractions
2. **Eliminates unnecessary dependencies** — Removes FastAPI, gymnasium, Redis, etc. (but may need Pydantic for data models)
3. **Enables system benchmarking** — MASEval evaluates agent _systems_, not just models
4. **Follows established precedent** — Similar to MACS benchmark, but with important differences (see below)

### Critical Difference from MACS

> **IMPORTANT**: Unlike MACS, tau2-bench tools have **REAL implementations with actual business logic**, not LLM simulations. The tools modify actual database state, and evaluation verifies the correctness of that state.

| Aspect              | MACS Benchmark                  | Tau2 Benchmark                      |
| ------------------- | ------------------------------- | ----------------------------------- |
| **Tools**           | LLM-simulated responses         | Real implementations that modify DB |
| **Evaluation**      | LLM-as-judge on assertions      | Deterministic DB state verification |
| **Reproducibility** | ±2-3% tolerance (LLM variance)  | Exact state matching required       |
| **Tool Porting**    | Not needed (generic simulation) | Must port domain-specific logic     |

### Rationale Summary

| Factor                | Re-implementation Advantage                                  |
| --------------------- | ------------------------------------------------------------ |
| **Dependencies**      | Removes ~20 unnecessary packages (keeps Pydantic for models) |
| **Architectural fit** | Native MASEval patterns                                      |
| **Flexibility**       | Can extend evaluation criteria                               |
| **Maintenance**       | Tool logic + domain data needs syncing                       |
| **Reproducibility**   | Validatable with contract tests (exact state matching)       |

Important: Existing code should be used as much as possible inline with this strategy. There is no value in reinventing the wheel unless we have to.

---

## 2. Architecture & Inversion of Control

### The Fundamental Inversion

**tau2-bench design**: Fixes the Agent implementation to evaluate different Models.

```
tau2-bench: Agent(fixed) × Model(variable) → Score
```

**MASEval design**: Fixes the Environment/Tasks to evaluate different Agent Systems.

```
MASEval: Environment(fixed) × AgentSystem(variable) → Score
```

This inversion is **critical** to the design. The benchmark must:

1. Provide abstract interfaces that accept _any_ agent implementation
2. Include a **Reference Agent** that replicates tau2-bench's original behavior for reproducibility validation

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Tau2Benchmark (Abstract)                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  User provides:                                                         ││
│  │  • setup_agents() → Their agent framework (smolagents, langgraph, etc.) ││
│  │  • get_model_adapter() → Their LLM provider                             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                     │                                        │
│                    ┌────────────────┼────────────────┐                      │
│                    ▼                ▼                ▼                      │
│           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│           │Tau2Environment│  │  Tau2User    │  │Tau2Evaluator │             │
│           │              │  │              │  │              │             │
│           │• Domain DB   │  │• User sim    │  │• DB state    │             │
│           │• Real tools  │  │• Multi-turn  │  │• Action eval │             │
│           │• Policies    │  │• Stop cond.  │  │• Assertions  │             │
│           │• get_db_hash │  │              │  │              │             │
│           └──────────────┘  └──────────────┘  └──────────────┘             │
│                    │                                                        │
│                    ▼                                                        │
│    ┌───────────────────────────────────────────────────────────┐           │
│    │              Domain-Specific Tools (Ported)               │           │
│    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │           │
│    │  │   Retail    │  │   Airline   │  │   Telecom   │       │           │
│    │  │  tools.py   │  │  tools.py   │  │  tools.py   │       │           │
│    │  │  models.py  │  │  models.py  │  │  models.py  │       │           │
│    │  └─────────────┘  └─────────────┘  └─────────────┘       │           │
│    │         ↓                ↓                ↓               │           │
│    │     RetailDB         AirlineDB        TelecomDB          │           │
│    │   (actual state)   (actual state)   (actual state)       │           │
│    └───────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Reference Agent for Reproducibility

To validate that MASEval's implementation produces equivalent results to upstream tau2-bench:

```python
class Tau2ReferenceAgent:
    """
    Reference agent that replicates tau2-bench's default LLMAgent behavior.

    Purpose:
    - Validation: Compare MASEval results with upstream tau2-bench
    - Baseline: Provide reproducible baseline for new agent architectures

    This agent is NOT intended for production use. Users should implement
    their own agents using setup_agents().
    """
```

The reference agent will:

1. Use the same prompts as tau2-bench's `LLMAgent`
2. Follow the same tool-calling patterns
3. Be used exclusively in validation/contract tests

---

## 3. Component Specifications

### 3.1 Data Loader (`data_loader.py`)

**Responsibility**: Download, parse, and provide access to tau2-bench domain data.

**Source Files to Adapt From**:

- `data/tau2/domains/{airline,retail,telecom}/tasks.json` → Task definitions
- `data/tau2/domains/{airline,retail,telecom}/db.json` → Environment state/data
- `data/tau2/domains/{airline,retail,telecom}/policy.md` → Policy constraints
- `data/tau2/domains/{airline,retail,telecom}/split_tasks.json` → Task splits (base/hard)

```python
# --- Interface Specification ---

VALID_DOMAINS: Tuple[str, ...] = ("airline", "retail", "telecom")
TASK_SPLITS: Tuple[str, ...] = ("base", "hard", "all")

def download_domain_data(
    data_dir: Optional[Path] = None,
    domain: Optional[str] = None,  # None = all domains
    verbose: int = 1,
) -> Path:
    """
    Download domain data from tau2-bench GitHub repository.

    Downloads: tasks.json, db.json, policy.md, split_tasks.json
    To: data_dir/original/{domain}/

    Returns: Path to original data directory

    # Adapted from: https://github.com/sierra-research/tau2-bench/tree/main/data/tau2/domains
    """

def load_tasks(
    domain: str,
    split: str = "base",
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> TaskCollection:
    """
    Load tasks for a tau2 domain.

    Args:
        domain: One of "airline", "retail", "telecom"
        split: One of "base", "hard", "all" (base recommended for reproducibility)
        data_dir: Base data directory
        limit: Maximum number of tasks

    Returns: TaskCollection with Task objects containing:
        - query: User's initial request
        - environment_data: Tools, database state, policies
        - evaluation_data: Assertions, expected outcomes
        - user_data: User profile, instructions
        - metadata: task_id, domain, split
    """

def load_domain_config(domain: str, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load domain configuration (policies, database schema, tool definitions).

    Returns: Dict with:
        - policy: Markdown policy text
        - database: Initial database state
        - tools: List of tool specifications
    """

def configure_model_ids(
    tasks: TaskCollection,
    *,
    tool_model_id: Optional[str] = None,
    user_model_id: Optional[str] = None,
    evaluator_model_id: Optional[str] = None,
) -> TaskCollection:
    """
    Configure model IDs for benchmark components.
    Same pattern as MACS benchmark.
    """
```

**Provenance Mapping**:
| MASEval Function | tau2-bench Source |
|------------------|-------------------|
| `download_domain_data()` | `data/tau2/domains/*/` |
| `load_tasks()` | `src/tau2/domains/{domain}/tasks.py` |
| `load_domain_config()` | `src/tau2/domains/{domain}/` |

---

### 3.2 Domain Tools (`domains/{retail,airline,telecom}/`)

**Responsibility**: Port actual tool implementations from tau2-bench that execute real business logic and modify database state.

> **CRITICAL**: Unlike MACS, these are NOT LLM-simulated. They contain real Python code that modifies database state. We must PORT this logic, not simulate it.

**Source Files to Port**:

- `src/tau2/domains/retail/tools.py` → `domains/retail/tools.py`
- `src/tau2/domains/airline/tools.py` → `domains/airline/tools.py`
- `src/tau2/domains/telecom/tools.py` → `domains/telecom/tools.py`

```python
# --- Interface Specification ---

class Tau2ToolBase(TraceableMixin):
    """
    Base class for ported tau2 tools.

    Unlike MACSGenericTool (LLM-simulated), these tools execute
    REAL business logic that modifies database state.

    # Adapted from: tau2-bench src/tau2/domains/*/tools.py
    """

    def __init__(self, db: "DomainDB"):
        """
        Args:
            db: Domain database instance (RetailDB, AirlineDB, etc.)
        """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def input_schema(self) -> Dict[str, Any]: ...

    def __call__(self, **kwargs) -> str:
        """
        Execute tool with REAL business logic.

        This actually modifies the database state, not a simulation.

        Raises:
            AgentError: Invalid arguments (agent's fault)
            EnvironmentError: Tool execution failure (not agent's fault)
        """

    def gather_traces(self) -> Dict[str, Any]: ...


# Example: Retail domain tool (ported from upstream)
class CancelPendingOrderTool(Tau2ToolBase):
    """
    Cancel a pending order and process refunds.

    # Adapted from: tau2-bench src/tau2/domains/retail/tools.py
    # Original function: cancel_pending_order()
    """

    def __call__(self, order_id: str, reason: str) -> str:
        """
        Actually cancels the order in RetailDB:
        - Updates order status to "cancelled"
        - Processes refund to original payment method
        - Updates gift card balance if applicable
        """
```

**Key Difference from MACS**:
| Aspect | MACS (MACSGenericTool) | Tau2 (Domain Tools) |
|--------|------------------------|---------------------|
| Execution | LLM generates response | Real Python code executes |
| State | No actual state changes | Modifies database state |
| Evaluation | LLM judges assertions | Checks actual DB state |
| Reproducibility | LLM variance (±2-3%) | Deterministic |

---

### 3.3 Tau2Environment (`environment.py`)

**Responsibility**: Create and manage tools for a domain, maintain **actual** database state, provide state verification methods.

**Reference**: `MACSEnvironment` pattern, but with critical additions for state verification.

```python
class Tau2Environment(Environment):
    """
    Environment for tau2 domains (airline, retail, telecom).

    Unlike MACSEnvironment, this manages REAL database state that
    tools actually modify. Provides methods for state verification.

    # Adapted from: tau2-bench src/tau2/environment/
    """

    def __init__(
        self,
        domain: str,
        task_data: Dict[str, Any],
    ):
        """
        Args:
            domain: One of "airline", "retail", "telecom"
            task_data: Task-specific environment data including initial DB state
        """

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize environment state from task data.

        State includes:
            - db: Domain database instance (RetailDB, AirlineDB, etc.)
            - policy: Domain policy text
            - initial_db_hash: Hash of initial state (for comparison)
        """

    def create_tools(self) -> Dict[str, Tau2ToolBase]:
        """
        Create all tools for this domain.

        Each tool receives a reference to the domain database.
        Tools execute real logic that modifies the database.
        """

    def get_db_hash(self) -> str:
        """
        Return hash of current database state.

        Used by evaluator to verify correct state changes.
        Critical for deterministic evaluation.

        # Adapted from: tau2-bench state verification
        """

    def check_assertions(self, assertions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run environment-specific assertion checks against current DB state.

        Args:
            assertions: List of assertion dicts from task evaluation_data

        Returns:
            List of assertion results with pass/fail status

        # Adapted from: tau2-bench src/tau2/evaluator/evaluator_env.py
        """

    def get_database_state(self) -> Dict[str, Any]:
        """Return current database state for evaluation/debugging."""

    def gather_traces(self) -> Dict[str, Any]:
        """
        Include database state changes in traces.

        Traces include:
            - initial_state_hash
            - final_state_hash
            - state_diff (what changed)
        """
```

**Provenance Mapping**:
| MASEval Component | tau2-bench Source |
|-------------------|-------------------|
| `Tau2Environment` | `src/tau2/environment/environment.py` |
| `get_db_hash()` | `src/tau2/environment/database.py` |
| `check_assertions()` | `src/tau2/evaluator/evaluator_env.py` |
| Domain databases | `src/tau2/domains/{domain}/db.py` |

---

### 3.4 Tau2User (`tau2.py`)

**Responsibility**: Multi-turn user simulation for interactive tasks.

**Reference**: `MACSUser` in `maseval/benchmark/macs/macs.py:447-583`

```python
class Tau2User(User):
    """
    User simulator for tau2 benchmark tasks.

    Simulates a user with specific goals interacting with an agent system.
    Supports multi-turn interaction with natural termination detection.

    # Adapted from: tau2-bench src/tau2/user/
    """

    DEFAULT_MAX_TURNS: int = 10  # tau2 allows more turns than MACS
    DEFAULT_STOP_TOKEN: str = "<task_complete>"

    def __init__(
        self,
        model: ModelAdapter,
        user_instructions: str,
        task_goal: str,
        initial_message: Optional[str] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        stop_token: str = DEFAULT_STOP_TOKEN,
    ):
        """
        Args:
            model: ModelAdapter for response generation
            user_instructions: Background and behavior instructions
            task_goal: What the user is trying to accomplish
            initial_message: First message to agent (if not generated)
            max_turns: Maximum interaction turns
            stop_token: Token indicating user satisfaction
        """

    @abstractmethod
    def get_tool(self) -> Any:
        """
        Return framework-specific tool for agent interaction.
        Must be implemented by framework-specific subclasses.
        """

    def simulate_response(self, agent_message: str) -> str:
        """Generate user response to agent message."""

    def is_done(self) -> bool:
        """Check if user goals are satisfied or max turns reached."""
```

**Key Differences from MACS**:

1. Higher default max turns (10 vs 5)
2. Different stop token format
3. Separate `user_instructions` and `task_goal` fields

---

### 3.5 Tau2Evaluator (`tau2.py`)

**Responsibility**: Evaluate task completion using multiple criteria.

**Reference**: `MACSEvaluator` in `maseval/benchmark/macs/macs.py:220-439`

```python
class Tau2Evaluator(Evaluator):
    """
    Evaluator for tau2 benchmark tasks.

    Combines multiple evaluation strategies:
    - Environment assertions (database state checks)
    - Action assertions (correct tool usage)
    - Communication assertions (appropriate responses)
    - NL assertions (natural language goal satisfaction)

    # Adapted from: tau2-bench src/tau2/evaluator/
    """

    def __init__(
        self,
        model: ModelAdapter,
        task: Task,
        environment: Tau2Environment,
        evaluation_types: List[str] = ["env", "action", "communicate"],
    ):
        """
        Args:
            model: ModelAdapter for NL assertion evaluation
            task: Task with evaluation_data containing assertions
            environment: Environment with database state
            evaluation_types: Which evaluators to run
        """

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant traces for evaluation."""

    def __call__(
        self,
        traces: Dict[str, Any],
        final_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate task completion.

        Returns:
            reward: Float [0.0, 1.0] - overall success metric
            reward_breakdown: Dict with per-evaluator scores
            assertions: List of individual assertion results
            passed: Boolean - did task pass all criteria?
        """

    def _evaluate_environment(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Check database state assertions."""

    def _evaluate_actions(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Check tool usage patterns."""

    def _evaluate_communication(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Check agent-user communication quality."""
```

**Provenance Mapping**:
| MASEval Method | tau2-bench Source |
|----------------|-------------------|
| `_evaluate_environment` | `src/tau2/evaluator/evaluator_env.py` |
| `_evaluate_actions` | `src/tau2/evaluator/evaluator_action.py` |
| `_evaluate_communication` | `src/tau2/evaluator/evaluator_communicate.py` |

---

### 3.6 Tau2Benchmark (`tau2.py`)

**Responsibility**: Orchestrate benchmark execution with user-provided agents.

**Reference**: `MACSBenchmark` in `maseval/benchmark/macs/macs.py:662-957`

```python
class Tau2Benchmark(Benchmark):
    """
    Tau 2 Benchmark - Framework-agnostic base class.

    Users subclass and implement:
    - setup_agents(): Create their agent system
    - get_model_adapter(): Provide LLM access

    The benchmark handles:
    - Environment setup with domain data
    - User simulation for multi-turn interaction
    - Multi-dimensional evaluation
    - Pass^k metric computation

    # Adapted from: tau2-bench src/tau2/run.py orchestration
    """

    def __init__(
        self,
        agent_data: Dict[str, Any],
        domain: str,
        callbacks: Optional[List[Any]] = None,
        n_task_repeats: int = 1,  # Set to k for Pass^k
        max_invocations: int = 10,
        **kwargs,
    ):
        """
        Args:
            agent_data: Agent configuration
            domain: One of "airline", "retail", "telecom"
            n_task_repeats: Repetitions per task (for Pass^k computation)
            max_invocations: Maximum agent-user exchange rounds
        """

    # --- Abstract methods (user must implement) ---

    @abstractmethod
    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """
        Create agent system for this task.

        This is the primary extension point. Users implement their
        agent architecture here (single agent, multi-agent, etc.).
        """

    @abstractmethod
    def get_model_adapter(self, model_id: str, **kwargs) -> ModelAdapter:
        """
        Provide ModelAdapter for benchmark components.

        Called to create adapters for:
        - Tool simulators
        - User simulator
        - Evaluators
        """

    # --- Provided implementations ---

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
    ) -> Tau2Environment:
        """Create environment with domain tools and database."""

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
    ) -> Tau2User:
        """Create user simulator from task data."""

    def setup_evaluators(
        self,
        environment: Tau2Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
    ) -> Sequence[Evaluator]:
        """Create evaluators for all evaluation dimensions."""

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Run all evaluators and aggregate results.

        Returns combined metrics including:
        - reward: Overall success [0.0, 1.0]
        - passed: Boolean task success
        - Per-dimension breakdowns
        """
```

---

### 3.7 Metrics Computation

**Responsibility**: Compute Pass^k and aggregate benchmark statistics.

```python
def compute_pass_at_k(
    results: List[Dict[str, Any]],
    k_values: List[int] = [1, 2, 3, 4],
) -> Dict[str, float]:
    """
    Compute Pass@k metrics from benchmark results.

    Pass@k: Probability that at least 1 of k attempts succeeds.

    Requires running benchmark with n_task_repeats >= max(k_values).

    Args:
        results: List of result dicts from benchmark.run()
        k_values: k values to compute (default: 1, 2, 3, 4 per tau2 paper)

    Returns:
        Dict with pass@1, pass@2, pass@3, pass@4 scores

    # Adapted from: tau2-bench metrics computation
    """

def compute_benchmark_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics across all results.

    Returns:
        - pass_at_k: Dict with pass@1 through pass@4
        - mean_reward: Average reward across tasks
        - success_rate: Proportion of passed tasks
        - per_domain_metrics: Breakdown by domain
        - status_counts: Execution status breakdown
    """
```

---

## 4. Provenance & Attribution Strategy

### File-Level Attribution

Every source file includes a module docstring with provenance:

```python
"""
Tau 2 Benchmark - [Component Name]

This module implements [description] for the tau2-bench benchmark.

Original benchmark: https://github.com/sierra-research/tau2-bench
Copyright (c) 2025 Sierra Research (MIT License)

Adapted components:
- [Tau2ClassName]: Adapted from src/tau2/[path]/[file].py
- [function_name]: Adapted from src/tau2/[path]/[file].py:[function]

MASEval-specific additions:
- [Feature]: [Description of MASEval-specific functionality]
"""
```

### Function-Level Attribution

Functions adapted from tau2-bench include inline comments:

```python
def _evaluate_environment(self, traces: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate environment state assertions.

    # Adapted from: tau2-bench src/tau2/evaluator/evaluator_env.py
    # Original function: EnvironmentEvaluator.calculate_reward()
    """
```

### Provenance Tracking Table

Maintain in `maseval/benchmark/tau2/PROVENANCE.md`:

| MASEval Component                     | tau2-bench Source                        | Adaptation Notes              |
| ------------------------------------- | ---------------------------------------- | ----------------------------- |
| `Tau2GenericTool`                     | `src/tau2/domains/*/tools.py`            | Added TraceableMixin          |
| `Tau2Environment`                     | `src/tau2/environment/`                  | Uses MASEval Environment base |
| `Tau2User`                            | `src/tau2/user/user.py`                  | Uses MASEval User base        |
| `Tau2Evaluator._evaluate_environment` | `src/tau2/evaluator/evaluator_env.py`    | Same logic, MASEval interface |
| `Tau2Evaluator._evaluate_actions`     | `src/tau2/evaluator/evaluator_action.py` | Same logic, MASEval interface |
| `compute_pass_at_k`                   | `src/tau2/metrics/`                      | Native implementation         |
| Domain data loading                   | `data/tau2/domains/`                     | Downloaded at runtime         |

---

## 5. Risk Assessment & Friction Points

### 5.1 State Divergence (CRITICAL)

**Risk**: Tool implementations must produce **exact same database state** as upstream. Any divergence causes evaluation to fail.

**Why This Is Critical**:

- Evaluation checks actual DB state, not LLM-judged assertions
- A single incorrect field value = task failure
- Different Pydantic versions could serialize differently

**Mitigation**:

1. **Contract tests** that compare DB state hash after tool sequences
2. Port tool logic **exactly** from upstream (minimize "improvements")
3. Use same Pydantic version as upstream or ensure serialization compatibility
4. Verify with: `assert our_db_hash == upstream_db_hash`

### 5.2 Pydantic Version Compatibility

**Risk**: tau2-bench uses Pydantic for data models. Version mismatches could cause:

- Different serialization behavior
- Different validation behavior
- Hash mismatches in state verification

**Mitigation**:

1. Check upstream Pydantic version requirement
2. Either match version or implement compatibility layer
3. Test serialization round-trips against upstream

### 5.3 Domain Tool Porting Complexity

**Risk**: Each domain has 10-30+ tools with complex business logic that must be ported accurately.

**Scope**:

- `retail/tools.py` — Order management, refunds, exchanges
- `airline/tools.py` — Booking, cancellation, seat changes
- `telecom/tools.py` — Plan changes, billing, support

**Mitigation**:

1. Port one domain at a time (start with `retail` as Gemini suggests)
2. Write unit tests for each tool against upstream expected outputs
3. Contract test: run tool sequence, compare final state

### 5.4 Hardcoded Prompts

**Risk**: tau2-bench has prompts embedded in multiple locations.

**Locations to Check**:

- `src/tau2/user/user.py` — User simulation prompts
- `src/tau2/agent/agent.py` — Agent system prompts (for reference agent)
- `src/tau2/evaluator/*` — NL evaluation prompts

**Mitigation**:

1. Extract all prompts to `prompt_templates/` directory
2. Document prompt sources in provenance
3. Allow runtime prompt customization via templates

### 5.5 Evaluation Logic Complexity

**Risk**: Multiple evaluation types with complex interaction.

**tau2-bench Evaluation Types**:

- `EnvironmentEvaluator` — Database state checks (DETERMINISTIC)
- `ActionEvaluator` — Tool usage validation (DETERMINISTIC)
- `CommunicateEvaluator` — Communication quality (may use LLM)
- `NLAssertionsEvaluator` — Natural language goal checking (uses LLM)

**Mitigation**:

1. Implement evaluators incrementally (env → action → communicate → NL)
2. Contract tests for each evaluator type
3. Start with `evaluation_types=["env", "action"]` for MVP (deterministic first)

### 5.6 Multi-Turn Conversation State

**Risk**: User simulation state must persist correctly across turns.

**Mitigation**:

1. Use MASEval's `User` base class (proven in MACS)
2. `Tau2User` manages conversation history
3. Stop token detection for natural termination

---

## 6. Implementation Phases

### Phase 1: Data Infrastructure (Est. 200 LOC)

**Goal**: Download and load tau2 domain data.

**Deliverables**:

```
maseval/benchmark/tau2/
├── __init__.py
├── loader.py
├── utils.py
└── data/
    └── .gitignore
```

**Validation**: Unit tests for data loading, schema validation.

---

### Phase 2: Retail Domain Porting (Est. 800 LOC)

**Goal**: Port retail domain as first complete implementation.

> **Why Retail First?** Gemini's plan suggests retail is simplest. Starting with one domain allows validating the entire architecture before scaling.

**Deliverables**:

```
maseval/benchmark/tau2/
├── domains/
│   └── retail/
│       ├── __init__.py
│       ├── tools.py      # Port ALL retail tools from upstream
│       ├── models.py     # Pydantic models (User, Order, Product, etc.)
│       └── db.py         # RetailDB with state management
├── environment.py        # Tau2Environment (retail-only initially)
└── evaluator.py          # Tau2Evaluator (env + action checks)
```

**Validation**:

- Unit tests for each retail tool
- **Contract test**: Run tool sequence, compare DB hash with upstream

---

### Phase 3: User & Benchmark Orchestration (Est. 400 LOC)

**Goal**: Complete benchmark flow with user simulation.

**Deliverables**:

```
maseval/benchmark/tau2/
├── user.py           # Tau2User with multi-turn support
├── benchmark.py      # Tau2Benchmark orchestration
├── agent.py          # Tau2ReferenceAgent (for validation)
└── prompt_templates/
    ├── user.txt
    └── evaluator.txt
```

**Validation**: End-to-end retail benchmark run with reference agent.

---

### Phase 4: Additional Domains (Est. 1200 LOC)

**Goal**: Port airline and telecom domains.

**Deliverables**:

```
maseval/benchmark/tau2/domains/
├── airline/
│   ├── tools.py, models.py, db.py
└── telecom/
    ├── tools.py, models.py, db.py
```

**Validation**: Contract tests for each domain against upstream.

---

### Phase 5: Framework Examples (Est. 400 LOC)

**Goal**: Demonstrate usage with smolagents and langgraph.

**Deliverables**:

```
examples/tau2_benchmark/
├── tau2_smolagents.py
├── tau2_langgraph.py
└── README.md
```

**Validation**: End-to-end execution on all domains.

---

### Phase 6: Validation & Documentation (Est. 200 LOC)

**Goal**: Ensure reproducibility against upstream, complete documentation.

**Deliverables**:

```
tests/test_benchmarks/test_tau2/
├── conftest.py
├── test_tau2_loader.py
├── test_tau2_environment.py
├── test_tau2_evaluator.py
├── test_tau2_benchmark.py
├── test_domains/
│   ├── test_retail_tools.py
│   ├── test_airline_tools.py
│   └── test_telecom_tools.py
└── test_tau2_contract.py      # DB state hash comparison

scripts/
└── validate_tau2_upstream.py  # Run both implementations, compare

docs/benchmark/tau2.md
PROVENANCE.md
```

**Validation**:

- Deterministic evaluators: **Exact** state hash match
- LLM-based evaluators: Within ±3% tolerance

---

## 7. File Structure Summary

```
maseval/benchmark/tau2/
├── __init__.py              # Exports, license attribution
├── loader.py                # Download, parse, load domain data
├── environment.py           # Tau2Environment with state verification
├── evaluator.py             # Tau2Evaluator (env, action, communicate checks)
├── user.py                  # Tau2User simulator
├── benchmark.py             # Tau2Benchmark orchestration
├── agent.py                 # Tau2ReferenceAgent (for validation)
├── utils.py                 # DB hashing, state diffing helpers
├── domains/                 # Domain-specific PORTED code
│   ├── __init__.py
│   ├── retail/
│   │   ├── __init__.py
│   │   ├── tools.py         # Ported from upstream retail/tools.py
│   │   ├── models.py        # Pydantic models for retail domain
│   │   └── db.py            # RetailDB class
│   ├── airline/
│   │   ├── __init__.py
│   │   ├── tools.py         # Ported from upstream airline/tools.py
│   │   ├── models.py        # Pydantic models for airline domain
│   │   └── db.py            # AirlineDB class
│   └── telecom/
│       ├── __init__.py
│       ├── tools.py         # Ported from upstream telecom/tools.py
│       ├── models.py        # Pydantic models for telecom domain
│       └── db.py            # TelecomDB class
├── prompt_templates/
│   ├── user.txt
│   └── evaluator.txt
├── data/
│   ├── .gitignore           # Ignore downloaded data
│   └── tasks/               # Downloaded task definitions
└── PROVENANCE.md            # Detailed source mapping

examples/tau2_benchmark/
├── tau2_smolagents.py
├── tau2_langgraph.py
└── README.md

tests/test_benchmarks/test_tau2/
├── conftest.py
├── test_tau2_loader.py
├── test_tau2_environment.py
├── test_tau2_user.py
├── test_tau2_evaluator.py
├── test_tau2_benchmark.py
├── test_domains/
│   ├── test_retail_tools.py    # Unit tests for each retail tool
│   ├── test_airline_tools.py   # Unit tests for each airline tool
│   └── test_telecom_tools.py   # Unit tests for each telecom tool
└── test_tau2_contract.py       # DB state hash comparison with upstream

docs/benchmark/tau2.md
```

---

## 8. Success Criteria

1. **Functional Completeness**

   - All three domains (airline, retail, telecom) supported
   - Pass^k metrics computed correctly
   - Multi-turn interaction works as expected

2. **Architectural Alignment**

   - Follows MASEval `Environment`, `Evaluator`, `Benchmark` patterns
   - Minimal additional dependencies (Pydantic for models)
   - Full callback/tracing integration

3. **Reproducibility** (CRITICAL)

   - **Deterministic evaluators** (env, action): Exact DB state hash match with upstream
   - **LLM-based evaluators** (communicate, NL): Within ±3% of upstream
   - Contract tests verify tool sequences produce identical state changes
   - Reference agent produces baseline results matching upstream

4. **Documentation**
   - API reference in docs
   - Usage examples for both frameworks
   - Provenance tracking complete
   - Clear documentation of what is ported vs. reimplemented

---

## Appendix: Reference Agent Specification

```python
class Tau2ReferenceAgent(AgentAdapter):
    """
    Reference implementation matching tau2-bench's LLMAgent.

    FOR VALIDATION ONLY. Not intended for production use.

    This agent replicates tau2-bench behavior to enable:
    1. Validation of MASEval implementation against upstream
    2. Baseline comparison for custom agent implementations

    # Adapted from: tau2-bench src/tau2/agent/agent.py LLMAgent
    """

    def __init__(
        self,
        model: ModelAdapter,
        tools: Dict[str, Tau2GenericTool],
        policy: str,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            model: LLM for agent reasoning
            tools: Available tools
            policy: Domain policy text
            system_prompt: Override default system prompt
        """

    def run(self, query: str) -> str:
        """
        Execute agent with tau2-bench-compatible behavior.

        Uses ReAct-style prompting matching upstream implementation.
        """
```

The reference agent enables running:

```bash
# Validation script
python scripts/validate_tau2_upstream.py --domain airline --tasks 10
# Compares: MASEval(ReferenceAgent) vs tau2-bench(LLMAgent)
# Asserts: Pass@1 difference < 3%
```
