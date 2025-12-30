from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterable, Optional, Sequence, Tuple, Union, cast
from datetime import datetime
import threading
from enum import Enum
import warnings

from .evaluator import Evaluator
from .task import Task, TaskCollection
from .environment import Environment
from .agent import AgentAdapter
from .model import ModelAdapter
from .callback_handler import CallbackHandler
from .callback import BenchmarkCallback
from .user import User
from .tracing import TraceableMixin
from .config import ConfigurableMixin
from .utils.system_info import gather_benchmark_config
from .callbacks.progress_bar import (
    ProgressBarCallback,
    TqdmProgressBarCallback,
    RichProgressBarCallback,
)
from .exceptions import AgentError, EnvironmentError, UserError


class TaskExecutionStatus(Enum):
    """Status of task execution and evaluation.

    This enum tracks the execution state of a task through the benchmark lifecycle,
    enabling graceful failure handling and comprehensive result reporting.

    The status distinguishes between errors caused by the agent (agent's fault) and
    errors caused by the evaluation infrastructure (environment, user simulator).
    This enables fair scoring by excluding infrastructure failures.

    Attributes:
        SUCCESS: Task executed and evaluated successfully.
        AGENT_ERROR: Agent violated contract at a boundary (agent's fault, counts against score).
        ENVIRONMENT_ERROR: Environment/tool infrastructure failed (not agent's fault, exclude from scoring).
        USER_ERROR: User simulator failed (not agent's fault, exclude from scoring).
        UNKNOWN_EXECUTION_ERROR: Unclassified execution error (e.g., agent framework internal failure).
        EVALUATION_FAILED: Task executed but evaluation raised an exception.
        SETUP_FAILED: Setup phase (environment, agents, evaluators) raised an exception.

    Scoring Guidance:
        - Include in agent score: SUCCESS, AGENT_ERROR
        - Exclude from agent score: ENVIRONMENT_ERROR, USER_ERROR, UNKNOWN_EXECUTION_ERROR
        - Handle separately: EVALUATION_FAILED, SETUP_FAILED
    """

    SUCCESS = "success"
    AGENT_ERROR = "agent_error"
    ENVIRONMENT_ERROR = "environment_error"
    USER_ERROR = "user_error"
    UNKNOWN_EXECUTION_ERROR = "unknown_execution_error"
    EVALUATION_FAILED = "evaluation_failed"
    SETUP_FAILED = "setup_failed"

    # Deprecated: kept for backward compatibility, use specific error types instead
    TASK_EXECUTION_FAILED = "task_execution_failed"


class Benchmark(ABC):
    """Abstract base class for orchestrating multi-agent system execution and evaluation.

    Benchmark provides a structured framework for running reproducible agent experiments across
    a collection of tasks. It manages the complete execution lifecycle: environment initialization,
    agent instantiation, task execution, and performance evaluation. The class enforces a
    three-stage pattern (setup, run, evaluate) while handling task iteration, callback orchestration,
    message history collection, and result aggregation.

    How to use:
        1. **Subclass Benchmark** and implement the abstract methods to define your benchmark logic
        2. **Implement setup methods** to specify how environments, agents, and evaluators are created
        3. **Implement run_agents** to define how your multi-agent system solves a single task
        4. **Instantiate your benchmark** with agent configuration
        5. **Call benchmark.run(tasks)** to execute the complete benchmark

        Example workflow:
            ```python
            class MyBenchmark(Benchmark):
                def setup_environment(self, agent_data, task):
                    return MyEnvironment(task.environment_data)

                def setup_agents(self, agent_data, environment, task, user):
                    agent = MyAgent(model=agent_data["model"])
                    agent_adapter = AgentAdapter(agent, "agent")
                    return [agent_adapter], {"agent": agent_adapter}

                def run_agents(self, agents, task, environment, query):
                    return agents[0].run(query)

                # ... implement other abstract methods

            # Run the benchmark
            benchmark = MyBenchmark(agent_data=config)
            reports = benchmark.run(tasks=my_tasks)

            # Retry failed tasks elegantly (graceful task failure handling by default)
            failed_tasks = benchmark.get_failed_tasks()
            if len(failed_tasks) > 0:
                retry_reports = benchmark.run(tasks=failed_tasks)

            # Or use strict mode for debugging (fail fast)
            benchmark = MyBenchmark(
                agent_data=config,
                fail_on_task_error=True,
                fail_on_evaluation_error=True,
                fail_on_setup_error=True
            )
            ```

        The framework handles task iteration, repetitions for statistical robustness, callback
        notifications, and result collection. You focus on defining agent behavior and evaluation
        criteria for your specific domain.
    """

    def __init__(
        self,
        agent_data: Optional[Dict[str, Any] | Iterable[Dict[str, Any]]] = None,
        callbacks: Optional[List[BenchmarkCallback]] = None,
        n_task_repeats: int = 1,
        max_invocations: int = 1,
        fail_on_setup_error: bool = False,
        fail_on_task_error: bool = False,
        fail_on_evaluation_error: bool = False,
        progress_bar: bool | str = True,
    ):
        """Initialize a benchmark with agent configurations.

        Args:
            agent_data: Configuration for agents. Either a single dict applied to all tasks, or
                an iterable of dicts with one configuration per task. If None, defaults to empty dict.
                Agent data typically includes model parameters, agent architecture details, and tool specifications.
            callbacks: Optional list of callback handlers for monitoring execution, tracing messages,
                or collecting custom metrics during the benchmark run.
            n_task_repeats: Number of times to repeat each task. Useful for measuring variance in
                stochastic agent behaviors. Must be at least 1.
            max_invocations: Maximum number of agent invocations per task in the execution loop.
                For simple benchmarks, the default (1) means agents run once per task. For interactive
                benchmarks with user feedback loops, set higher (e.g., 5 for MACS) to allow multiple
                agent-user interaction rounds.
            fail_on_setup_error: If True, raise exceptions when setup fails (environment, agents, evaluators).
                If False (default), catch exceptions during setup and record them in the report with status
                SETUP_FAILED. This allows the benchmark to continue running remaining tasks even if setup fails.
            fail_on_task_error: If True, raise exceptions when task execution fails. If False (default),
                catch exceptions during task execution and record them in the report with status
                TASK_EXECUTION_FAILED. This allows the benchmark to continue running remaining tasks.
            fail_on_evaluation_error: If True, raise exceptions when evaluation fails. If False (default),
                catch exceptions during evaluation and record them in the report with status
                EVALUATION_FAILED. This allows the benchmark to continue even if evaluation logic has errors.
            progress_bar: Controls progress bar during benchmark runs. When enabled, one of the
                ProgressBarCallback will be added. Options:
                - True (default): Automatically adds a TqdmProgressBarCallback if no ProgressBarCallback
                  is already present in the `callbacks` argument.
                - False: Does not automatically add a progress bar callback. Manually provided
                  ProgressBarCallback instances in the callbacks list will still work normally.
                - "tqdm": Automatically adds a TqdmProgressBarCallback (same as True)
                - "rich": Automatically adds a RichProgressBarCallback (uses the rich library)

        Raises:
            ValueError: If n_task_repeats is less than 1.

        How to use:
            Provide either a single agent configuration for all tasks, or task-specific configurations:

            ```python
            # Single config for all tasks
            benchmark = MyBenchmark(agent_data={"model": "gpt-4", "temperature": 0.7})

            # Task-specific configs (will be validated in run() based on task count)
            benchmark = MyBenchmark(
                agent_data=[
                    {"model": "gpt-4", "config": "easy"},
                    {"model": "gpt-4", "config": "hard"}
                ]
            )

            # Enable failure-safe execution (default behavior)
            benchmark = MyBenchmark(
                agent_data=config,
                fail_on_task_error=False,  # Continue on task failures
                fail_on_evaluation_error=False  # Continue on evaluation failures
            )

            # Strict mode - fail fast on any error (useful for debugging)
            benchmark = MyBenchmark(
                agent_data=config,
                fail_on_task_error=True,
                fail_on_evaluation_error=True,
                fail_on_setup_error=True
            )

            # Progress bar configuration (automatically adds a callback)
            benchmark = MyBenchmark(agent_data=config)  # Default: adds TqdmProgressBarCallback
            benchmark = MyBenchmark(agent_data=config, progress_bar=True)  # Explicit: TqdmProgressBarCallback
            benchmark = MyBenchmark(agent_data=config, progress_bar="tqdm")  # Same as True
            benchmark = MyBenchmark(agent_data=config, progress_bar="rich")  # Uses RichProgressBarCallback
            benchmark = MyBenchmark(agent_data=config, progress_bar=False)  # No automatic callback

            # Progress bar configuration (manually add a callback)
            benchmark = MyBenchmark(
                agent_data=config,
                callbacks=[MyCustomProgressBarCallback()]  # User-defined progress bar
            )
            ```
        """
        # Store agent_data, defaulting to empty dict if None
        self.agent_data = agent_data if agent_data is not None else {}

        # Initialize tasks to empty collection (will be set in run())
        self.tasks = TaskCollection([])

        self.callback_handler = CallbackHandler()
        self.callbacks = callbacks or []

        # Attach default progress bar if enabled and no progress bar in callbacks
        if progress_bar is not False:
            # Check if user already provided a progress bar callback
            has_progress_bar = any(isinstance(cb, ProgressBarCallback) for cb in self.callbacks)

            if not has_progress_bar:
                # Attach default progress bar based on type
                if progress_bar is True or progress_bar == "tqdm":
                    self.callbacks.append(TqdmProgressBarCallback())
                elif progress_bar == "rich":
                    self.callbacks.append(RichProgressBarCallback())
                else:
                    raise ValueError(f"Invalid progress_bar value: {progress_bar!r}. Must be True, False, 'tqdm', or 'rich'.")

        self.n_task_repeats = n_task_repeats
        if self.n_task_repeats < 1:
            raise ValueError("n_task_repeats must be at least 1")

        # Execution loop configuration
        self.max_invocations = max_invocations

        # Failure handling configuration
        self.fail_on_task_error = fail_on_task_error
        self.fail_on_evaluation_error = fail_on_evaluation_error
        self.fail_on_setup_error = fail_on_setup_error

        # Registry for Traceable components (cleared after each task repetition)
        self._trace_registry: Dict[str, TraceableMixin] = {}
        self._component_id_map: Dict[int, str] = {}  # Maps id(component) -> registry key

        # Registry for Configurable components (cleared after each task repetition)
        self._config_registry: Dict[str, ConfigurableMixin] = {}
        self._config_component_id_map: Dict[int, str] = {}  # Maps id(component) -> registry key

        # Persistent benchmark-level reports (stored across all task repetitions)
        # Each report contains both traces and configs for a single task repetition
        self.reports: List[Dict[str, Any]] = []

        # Gather benchmark-level configuration (system, git, packages, etc.)
        self.benchmark_level_config = gather_benchmark_config()

    def register(self, category: str, name: str, component: TraceableMixin) -> TraceableMixin:
        """Register a component for comprehensive trace and configuration collection.

        All core MASEval components (AgentAdapter, ModelAdapter, Environment,
        User, LLMSimulator, BenchmarkCallback) inherit from TraceableMixin and
        ConfigurableMixin, and are automatically registered for both trace and
        configuration collection before evaluation.

        **Note:** Most components are **automatically registered** when returned from
        setup methods (`setup_environment`, `setup_user`, `setup_agents`). You only
        need to manually register additional components like models, simulators, or
        tools that aren't automatically captured.

        Args:
            category: Component category (e.g., "agents", "models", "tools", "simulators",
                     "callbacks", "user", "environment"). Use plural form to match
                     the structure in collect_all_traces() and collect_all_configs().
            name: Unique identifier for this component within its category
            component: Any object inheriting from TraceableMixin (and optionally ConfigurableMixin)

        Returns:
            The component (for chaining convenience)

        Raises:
            ValueError: If the component is already registered under a different name

        How to use:
            Most components are auto-registered. Manual registration is only needed
            for additional components:

            ```python
            def setup_agents(self, agent_data, environment, task, user):
                # Create model (needs manual registration)
                model = MyModelAdapter(...)
                self.register("models", "main_model", model)

                # Create agent (auto-registered when returned)
                agent = MyAgent(model=model)
                agent_adapter = AgentAdapter(agent, "agent1")

                # Environment and user are also auto-registered
                return [agent_adapter], {"agent1": agent_adapter}
            ```

            Traces and configs are automatically collected before evaluation via
            `collect_all_traces()` and `collect_all_configs()` which are called
            internally by the `run()` method.
        """
        # Check if this component is already registered for traces
        component_id = id(component)
        if component_id in self._component_id_map:
            existing_key = self._component_id_map[component_id]
            existing_category, existing_name = existing_key.split(":", 1)
            new_key = f"{category}:{name}"

            if existing_key == new_key:
                # Same component, same name - silently accept (idempotent)
                return component
            else:
                raise ValueError(
                    f"Component is already registered as '{existing_key}' and cannot be "
                    f"re-registered as '{new_key}'. Note: Environments, users, and agents "
                    f"returned from setup methods are automatically registered."
                )

        key = f"{category}:{name}"

        # Register for trace collection
        self._trace_registry[key] = component
        self._component_id_map[component_id] = key

        # Also register for configuration collection if component supports it
        if isinstance(component, ConfigurableMixin):
            self._config_registry[key] = component
            self._config_component_id_map[component_id] = key

        return component

    def clear_registry(self) -> None:
        """Clear the component registry after a task repetition completes.

        This method is called automatically by `run()` after each task repetition
        to ensure components are not carried over between repetitions. The
        reports list persists across all repetitions for aggregated analysis.
        """
        self._trace_registry.clear()
        self._component_id_map.clear()
        self._config_registry.clear()
        self._config_component_id_map.clear()

    def collect_all_traces(self) -> Dict[str, Any]:
        """Collect execution traces from all registered components for the current task repetition.

        This method is called automatically by `run()` after each task repetition completes
        and before evaluation begins. It gathers comprehensive traces from all registered
        components (agents, models, tools, simulators, callbacks, etc.) for that specific
        repetition. After collection, the registry is cleared for the next repetition.

        The collected traces are stored in `benchmark.reports` list along with configs
        for persistent access across all task repetitions.

        Returns:
            Structured dictionary containing:
            - metadata: Collection timestamp and thread info
            - agents: Dict mapping agent names to their traces (messages, execution data)
            - models: Dict mapping model names to their traces (API calls, timing, errors)
            - tools: Dict mapping tool names to their traces (invocations, parameters)
            - simulators: Dict mapping simulator names to their traces (attempts, outcomes)
            - callbacks: Dict mapping callback names to their traces (custom data)
            - environment: Direct traces from the environment (not nested), or None if not present
            - user: Direct traces from the user simulator (not nested), or None if not present
            - other: Dict for any other registered components

        How to use:
            This method is called automatically by `run()` after each task repetition:

            ```python
            # Automatic collection (recommended)
            results = benchmark.run()

            # Access all collected reports (traces + configs) across repetitions
            for report in benchmark.reports:
                print(f"Task {report['task_id']}, Repeat {report['repeat_idx']}")
                # Agents is a dict: agent_name -> traces
                print(f"Agent messages: {report['traces']['agents']['my_agent']}")
                # Environment and user are direct (not nested)
                print(f"Environment state: {report['traces']['environment']}")
                print(f"User interactions: {report['traces']['user']}")
            ```

            The collected traces are passed to the evaluator's `evaluate()` method
            and stored in `benchmark.reports` for later analysis.
        """
        traces: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "thread_id": threading.current_thread().ident,
                "total_components": len(self._trace_registry),
            },
            "agents": {},
            "models": {},
            "tools": {},
            "simulators": {},
            "callbacks": {},
            "environment": None,
            "user": None,
            "other": {},
        }

        for key, component in self._trace_registry.items():
            category, comp_name = key.split(":", 1)

            try:
                component_traces = component.gather_traces()

                # Inject name from registry if component doesn't have it
                # Is this intervention obfuscating the mechnaisms too much?
                if "name" not in component_traces:
                    component_traces["name"] = comp_name

                # Handle environment and user as direct values (not nested in dict)
                if category == "environment":
                    traces["environment"] = component_traces
                elif category == "user":
                    traces["user"] = component_traces
                else:
                    # Ensure category exists in traces
                    if category not in traces:
                        traces[category] = {}
                    traces[category][comp_name] = component_traces
            except Exception as e:
                # Gracefully handle tracing errors
                error_info = {
                    "error": f"Failed to gather traces: {e}",
                    "error_type": type(e).__name__,
                    "component_type": type(component).__name__,
                }

                if category == "environment":
                    traces["environment"] = error_info
                elif category == "user":
                    traces["user"] = error_info
                else:
                    if category not in traces:
                        traces[category] = {}
                    traces[category][comp_name] = error_info

        return traces

    def collect_all_configs(self) -> Dict[str, Any]:
        """Collect configuration from all registered components for the current task repetition.

        This method is called automatically by `run()` after each task repetition completes
        and before evaluation begins. It gathers comprehensive configuration from all registered
        components (agents, models, tools, simulators, callbacks, etc.) for that specific
        repetition. After collection, the registry is cleared for the next repetition.

        The collected configs are stored in `benchmark.reports` list along with traces
        for persistent access across all task repetitions.

        Returns:
            Structured dictionary containing:
            - metadata: Collection timestamp and thread info
            - agents: Dict mapping agent names to their config (settings, parameters)
            - models: Dict mapping model names to their config (model IDs, parameters)
            - tools: Dict mapping tool names to their config (specifications, settings)
            - simulators: Dict mapping simulator names to their config (parameters, templates)
            - callbacks: Dict mapping callback names to their config (settings)
            - environment: Direct config from the environment (not nested), or None if not present
            - user: Direct config from the user simulator (not nested), or None if not present
            - other: Dict for any other registered components
            - benchmark: Benchmark-level configuration (git, system, packages)

        How to use:
            This method is called automatically by `run()` after each task repetition:

            ```python
            # Automatic collection (recommended)
            results = benchmark.run()

            # Access all collected reports (traces + configs) across repetitions
            for report in benchmark.reports:
                print(f"Task {report['task_id']}, Repeat {report['repeat_idx']}")
                # Agents is a dict: agent_name -> config
                print(f"Agent config: {report['config']['agents']['my_agent']}")
                # Environment and user are direct (not nested)
                print(f"Environment config: {report['config']['environment']}")
                print(f"User config: {report['config']['user']}")
                # Benchmark-level config
                print(f"Git commit: {report['config']['benchmark']['git']['commit_hash']}")
            ```

            The collected configs are available in the results for reproducibility analysis.
        """
        configs: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "thread_id": threading.current_thread().ident,
                "total_components": len(self._config_registry),
            },
            "agents": {},
            "models": {},
            "tools": {},
            "simulators": {},
            "callbacks": {},
            "environment": None,
            "user": None,
            "other": {},
            "benchmark": self.benchmark_level_config,  # Include benchmark-level config
        }

        for key, component in self._config_registry.items():
            category, comp_name = key.split(":", 1)

            try:
                component_config = component.gather_config()

                # Inject name from registry if component doesn't have it
                # Is this intervention obfuscating the mechnaisms too much?
                if "name" not in component_config:
                    component_config["name"] = comp_name

                # Handle environment and user as direct values (not nested in dict)
                if category == "environment":
                    configs["environment"] = component_config
                elif category == "user":
                    configs["user"] = component_config
                else:
                    # Ensure category exists in configs
                    if category not in configs:
                        configs[category] = {}
                    configs[category][comp_name] = component_config
            except Exception as e:
                # Gracefully handle config gathering errors
                error_info = {
                    "error": f"Failed to gather config: {e}",
                    "error_type": type(e).__name__,
                    "component_type": type(component).__name__,
                }

                if category == "environment":
                    configs["environment"] = error_info
                elif category == "user":
                    configs["user"] = error_info
                else:
                    if category not in configs:
                        configs[category] = {}
                    configs[category][comp_name] = error_info

        return configs

    def add_callback(self, callback: BenchmarkCallback) -> None:
        """Register a callback handler to monitor benchmark execution.

        Args:
            callback: A BenchmarkCallback instance that will receive execution events.

        How to use:
            Callbacks receive notifications at key lifecycle points for tracing, progress tracking,
            or custom metrics collection. See [`BenchmarkCallback`][maseval.core.callback.BenchmarkCallback]
            for available hooks and their signatures.

            ```python
            from maseval.core.callbacks import MessageTracingCallback

            benchmark = MyBenchmark(tasks=tasks, agent_data=config)
            benchmark.add_callback(MessageTracingCallback(output_dir="logs"))
            results = benchmark.run()
            ```
        """
        self.callback_handler.register(callback.on_event)

    @abstractmethod
    def setup_environment(self, agent_data: Dict[str, Any], task: Task) -> Environment:
        """Create and initialize the environment for a task.

        This method is called once per task execution to create a fresh environment with the
        task's initial conditions.

        **Note:** The returned environment is **automatically registered** for tracing.
        You don't need to manually call `register()` for it.

        Args:
            agent_data: Configuration dict containing agent specifications, model parameters,
                and other settings that may influence environment setup (e.g., framework type).
            task: The Task object containing environment_data, query, and metadata needed to
                construct the environment.

        Returns:
            An Environment instance with initialized state and tools for this specific task.

        How to use:
            The environment encapsulates task state and provides tools/APIs that agents can use.
            Your implementation should:

            - Extract environment state from task.environment_data
            - Initialize any databases, simulators, or API clients
            - Create and configure tools that agents can invoke
            - Set up domain-specific state (inventory, user profiles, etc.)
            - Optionally use agent_data for framework-specific tool initialization

            ```python
            def setup_environment(self, agent_data: Dict[str, Any], task: Task) -> Environment:
                # Extract data and initialize
                framework = agent_data.get("framework", "default")
                return TravelEnvironment(
                    hotels=task.environment_data["hotels"],
                    flights=task.environment_data["flights"],
                    user_preferences=task.environment_data.get("preferences", {}),
                    framework=framework  # For framework-specific tool creation
                )
            ```

            The environment is automatically registered for tracing when returned.
        """
        pass

    def setup_user(self, agent_data: Dict[str, Any], environment: Environment, task: Task) -> Optional[User]:
        """Create an optional user simulator for interactive tasks.

        This method is optional. Return None if your benchmark does not require user simulation.

        **Note:** The returned user is **automatically registered** for tracing.
        You don't need to manually call `register()` for it.

        Args:
            agent_data: Configuration dict containing agent specifications and settings that may
                influence user simulator setup (e.g., framework type for creating compatible tools).
            environment: The environment instance created for this task.
            task: The Task object with user profile data or scenario information.

        Returns:
            A User instance that can respond to agent queries, or None if not needed.

        How to use:
            User simulators enable agent-user interactions by responding to queries with preferences,
            clarifications, or feedback. Useful for benchmarks testing conversational agents or
            systems requiring user input during execution.

            ```python
            def setup_user(self, agent_data: Dict[str, Any], environment: Environment, task: Task) -> User:
                framework = agent_data.get("framework", "default")
                return UserSimulator(
                    profile=task.environment_data.get("user_profile"),
                    scenario=task.metadata["scenario"],
                    model=self.user_model,
                    framework=framework  # For framework-specific tool creation
                )

            # Or skip user simulation entirely
            def setup_user(self, agent_data: Dict[str, Any], environment: Environment, task: Task) -> None:
                return None
            ```

            The user is automatically registered for tracing when returned.
        """
        pass

    @abstractmethod
    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Instantiate and configure the agent system for a task.

        **Note:** All agents in the returned `agents_dict` are **automatically registered**
        for tracing. You don't need to manually call `register()` for them. However, you
        should manually register models, simulators, or other components used by agents.

        Args:
            agent_data: Configuration dict containing agent specifications, model parameters, and
                tool assignments for this task.
            environment: The initialized environment providing tools to the agents.
            task: The Task object with query and metadata.
            user: Optional user simulator for agent-user interactions.

        Returns:
            A tuple of (agents_to_run, agents_dict) where:

            - agents_to_run: Sequence of agents to invoke in run_agents() (typically 1 orchestrator)
            - agents_dict: Dictionary mapping agent names/IDs to all agent instances for monitoring

        How to use:
            This method constructs your agent architecture—single agent, multiple collaborative agents,
            or an orchestrator managing workers. Each agent is wrapped in AgentAdapter for uniform
            message history tracking.

            The dual return structure serves different purposes:

            - **agents_to_run**: Only agents directly invoked in run_agents() (typically the orchestrator)
            - **agents_dict**: All agents in the system for message history collection from workers
              called indirectly through the orchestrator

            ```python
            def setup_agents(self, agent_data, environment, task, user):
                # Manually register model (not auto-registered)
                model = MyModelAdapter(...)
                self.register("model", "main_model", model)

                # Create worker agents
                workers = {}
                for spec in agent_data["workers"]:
                    agent = ToolAgent(model=model, tools=environment.get_tools(spec["tools"]))
                    workers[spec["id"]] = AgentAdapter(agent, spec["id"])

                # Create orchestrator with access to workers
                orchestrator = OrchestratorAgent(
                    model=model,
                    managed_agents=[w.agent for w in workers.values()]
                )
                orchestrator_adapter = AgentAdapter(orchestrator, "orchestrator")

                # Return orchestrator to run, but all agents for monitoring
                # All agents auto-registered for tracing
                all_agents = {"orchestrator": orchestrator_adapter, **workers}
                return [orchestrator_adapter], all_agents
            ```
        """
        pass

    @abstractmethod
    def setup_evaluators(
        self,
        environment: Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
    ) -> Sequence[Evaluator]:
        """Create evaluators to assess agent performance on a task.

        Args:
            environment: The environment instance with final state after agent execution.
            task: The Task object with evaluation criteria in evaluation_data.
            agents: The agents that will execute the task (useful for context in evaluation).
            user: Optional user simulator, if evaluation needs user interaction data.

        Returns:
            A sequence of Evaluator instances that will be called with execution traces.

        How to use:
            Evaluators judge whether agents successfully completed the task or satisfied specific
            criteria. Multiple evaluators can measure different performance aspects (accuracy,
            efficiency, conversation quality, etc.).

            ```python
            def setup_evaluators(self, environment, task, agents, user):
                return [
                    SuccessEvaluator(task.evaluation_data["gold_answer"]),
                    EfficiencyEvaluator(max_steps=task.metadata["max_steps"]),
                    LLMJudgeEvaluator(model=self.judge_model, criteria=task.evaluation_data["rubric"])
                ]
            ```
        """
        pass

    @abstractmethod
    def get_model_adapter(self, model_id: str, **kwargs) -> ModelAdapter:
        """Provide a ModelAdapter for benchmark components that require LLM access.

        Many benchmark components beyond the agents themselves require access to language
        models. Common examples include:

        - **Tool simulators**: Simulating tool responses when real APIs aren't available
        - **User simulators**: Generating realistic user responses in multi-turn dialogues
        - **Judges/Evaluators**: Using LLMs to assess agent performance against criteria
        - **Reward models**: Computing scores for reinforcement learning

        This method centralizes model provisioning, giving you control over which models
        are used throughout the benchmark. Implement this to return a configured ModelAdapter
        for the requested model.

        Args:
            model_id: The model identifier to use (e.g., "gemini-2.5-flash",
                "openrouter/google/gemini-2.5-flash", "gpt-4o"). This is passed by the
                benchmark when setting up components that need model access.
            **kwargs: Additional arguments for adapter creation or registration. Common kwargs:
                - register_category: Category for trace registration (e.g., "models")
                - register_name: Name for trace registration (e.g., "evaluator_user_gsr")

        Returns:
            A ModelAdapter instance configured for the specified model. For proper tracing,
            return a fresh adapter for each call rather than reusing instances. You can
            still share the underlying API client for efficiency.

        How to use:
            For proper tracing, register the adapter after creation using the kwargs:

            ```python
            def get_model_adapter(self, model_id: str, **kwargs) -> ModelAdapter:
                adapter = GoogleGenAIModelAdapter(self.client, model_id=model_id)

                # Register for tracing if registration info provided
                category = kwargs.get("register_category", "models")
                name = kwargs.get("register_name", model_id)
                self.register(category, name, adapter)

                return adapter
            ```

            The benchmark calls this method when setting up tools, user simulators,
            and evaluators. Each call creates a fresh adapter with its own trace log.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Execute evaluators to score agent performance on the task.

        This method calls each evaluator with the collected execution data to produce evaluation
        results. The framework automatically collects comprehensive execution traces from all
        registered components before calling this method.

        Args:
            evaluators: The evaluator instances created by setup_evaluators().
            agents: Dictionary of all agent instances, useful if evaluators need to query agent
                state beyond message history.
            final_answer: The return value from run_agents() - typically the final output, answer,
                or result from the agent system's execution.
            traces: Comprehensive execution traces from all registered components including:
                - agents: Dict mapping agent names to their traces (messages, execution metadata)
                - models: Dict mapping model names to traces (API calls, timing, token usage, errors)
                - tools: Dict mapping tool names to traces (invocations with parameters and results)
                - simulators: Dict mapping simulator names to traces (attempts and outcomes)
                - callbacks: Dict mapping callback names to traces (custom callback data)
                - environment: Direct environment traces (state, tools) - not nested in a dict
                - user: Direct user simulator traces (interactions) - not nested in a dict
                - metadata: Collection timestamp and thread info
                Plus any other registered component categories.

        Returns:
            List of evaluation result dictionaries, typically one per evaluator. Each dict should
            contain metrics, scores, or judgments about the agent's performance.

        How to use:
            Evaluators can access comprehensive execution data including final answers and traces:

            ```python
            def evaluate(self, evaluators, agents, final_answer, traces):
                results = []
                for evaluator in evaluators:
                    # Access agent message histories from traces
                    agent_traces = {name: trace_data.get('messages', MessageHistory())
                                   for name, trace_data in traces.get('agents', {}).items()}

                    # Access additional execution data
                    model_traces = traces.get("models", {})
                    tool_traces = traces.get("tools", {})
                    environment_traces = traces.get("environment")  # Direct, not nested
                    user_traces = traces.get("user")  # Direct, not nested

                    # Pass comprehensive data to evaluator
                    result = evaluator(
                        final_answer=final_answer,
                        traces=agent_traces,
                        model_calls=model_traces,
                        tool_usage=tool_traces,
                        environment_state=environment_traces,
                        user_interactions=user_traces
                    )
                    results.append(result)
                return results
            ```
        """
        pass

    @abstractmethod
    def run_agents(self, agents: Sequence[AgentAdapter], task: Task, environment: Environment, query: str) -> Any:
        """Execute the agent system to solve a single task instance.

        This method is called once per task repetition by the framework's `run()` loop.

        Args:
            agents: Sequence of agents to execute (typically just the orchestrator or main agent).
            task: The Task object with the query and any metadata needed for execution.
            environment: The environment instance providing tools and state.
            query: The query string to pass to agents. For single-turn benchmarks this is
                typically task.query. For multi-turn with users, this may be an initial
                prompt or simulated user response.

        Returns:
            The final answer or result from the agent system's execution. This could be:
            - A string containing the final answer
            - A dict with structured output
            - A list of answers from multiple agents
            - Any other representation of the task solution

            Note: Message traces are captured automatically through the tracing system and
            passed to evaluate() via the traces parameter. Do NOT return message histories here.

        How to use:
            This is where you implement your multi-agent system logic for solving one task, one time.
            You define how your agents interact, communicate, and collaborate to complete the task.

            The agents parameter contains only the primary agents to run (from agents_to_run in
            setup_agents). Worker agents called indirectly through an orchestrator do not appear here.

            Note: This method handles a single task execution. The framework's `run()` method manages
            task iteration, repetitions, and the complete benchmark lifecycle.

            ```python
            def run_agents(self, agents, task, environment, query):
                # Simple single-agent execution - returns final answer string
                orchestrator = agents[0]
                final_answer = orchestrator.run(query)
                return final_answer

            # Or for multiple agents returning a list of answers:
            def run_agents(self, agents, task, environment, query):
                answers = []
                for agent in agents:
                    answer = agent.run(query)
                    answers.append(answer)
                return answers
            ```
        """
        pass

    def execution_loop(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Environment,
        user: Optional[User],
    ) -> Any:
        """Execute agents with optional user interaction loop.

        This method orchestrates the agent-user interaction pattern. When a user is
        present, the user initiates the conversation using `User.get_intial_query`.
        If no user is present, ``task.query`` is used as the initial query.

        Interaction Flow:
            By default, agents execute once (``max_invocations=1``). For multi-turn
            interaction, set ``self.max_invocations > 1`` in your benchmark's ``__init__``.
            The loop continues until ``max_invocations`` is reached or ``user.is_done()``
            returns True (e.g., max turns reached or stop token detected).

        Note:
            Override this method in your benchmark subclass to implement custom
            interaction patterns (e.g., agent-initiated conversations, different
            termination conditions, or specialized query routing).

        Args:
            agents: Agents to execute (typically the orchestrator).
            task: The task being solved.
            environment: The environment providing tools and state.
            user: Optional user simulator. If provided, the user initiates and drives
                the conversation. If None, a single agent execution with ``task.query``.

        Returns:
            Final answer from the last agent execution.

        Example:
            For interactive benchmarks, enable multi-turn interaction::

                def __init__(self, ...):
                    super().__init__(...)
                    self.max_invocations = 5  # Up to 5 agent-user exchanges
        """

        final_answer = None

        # Determine initial query text
        if user is not None:
            query_text = user.get_initial_query()
        else:
            query_text = task.query

        for _ in range(self.max_invocations):
            # Execute agents with query
            final_answer = self.run_agents(agents, task, environment, query_text)

            # No user means single execution
            if user is None:
                break

            # Simulate user response (handles message recording, stop token detection, turn counting)
            user_response = user.simulate_response(str(final_answer) if final_answer else "")

            # Check if user is done (cheap state check - no LLM call)
            if user.is_done():
                break

            # Use user's response as next query
            query_text = user_response

        return final_answer

    def run(self, tasks: Union[Task, TaskCollection, Iterable[Union[Task, dict]]]) -> List[Dict[str, Any]]:
        """Initialize and execute the complete benchmark loop across all tasks.

        Args:
            tasks: Collection of tasks to execute. Can be a single Task, TaskCollection,
                list of Task objects, or list of dicts that will be converted to Tasks.

        Returns:
            List of report dictionaries, one per task repetition. Each report contains:
            - task_id: Task identifier (UUID)
            - repeat_idx: Repetition index (0 to n_task_repeats-1)
            - status: Execution status (one of TaskExecutionStatus enum values)
            - traces: Execution traces from all registered components
            - config: Configuration from all registered components and benchmark level
            - eval: Evaluation results (None if task or evaluation failed)
            - error: Error details dict (only present if status is not SUCCESS), containing:
                - error_type: Exception class name
                - error_message: Exception message
                - traceback: Full traceback string

        Raises:
            ValueError: If agent_data length doesn't match number of tasks (when agent_data is an iterable).

        How to use:
            This is the framework's main orchestration method that runs your entire benchmark. It
            iterates through all tasks, handles repetitions, and manages the three-stage lifecycle
            for each execution. You don't implement this method—instead, you call it to start the
            benchmark after implementing the setup and execution methods.

            By default, the benchmark will continue executing remaining tasks even if some fail.
            You can change this behavior by setting `fail_on_task_error=True`,
            `fail_on_evaluation_error=True`, or `fail_on_setup_error=True` when instantiating
            the benchmark. Each task execution returns a status indicating success or the specific
            failure type (see [`TaskExecutionStatus`][maseval.core.benchmark.TaskExecutionStatus]).

            For each task execution, the framework:

            1. Calls your setup methods to initialize components
            2. Calls your `run_agents()` method to execute the task
            3. Collects message histories and calls evaluators
            4. Stores results and triggers callbacks

            Pseudocode structure:
                ```
                for task in tasks:
                    for repeat in range(n_task_repeats):
                        # Setup stage
                        environment = setup_environment(agent_data, task)
                        user = setup_user(agent_data, environment, task)
                        agents_to_run, agents_dict = setup_agents(agent_data, environment, task, user)
                        evaluators = setup_evaluators(environment, task, agents_to_run, user)

                        # Run stage (execution_loop handles multi-turn if user exists)
                        agents_output = execution_loop(agents_to_run, task, environment, user)

                        # Evaluate stage
                        traces = collect_message_histories(agents_dict)
                        eval_results = evaluate(evaluators, traces, agents_dict)

                        # Store results
                        store_result(task_id, traces, eval_results)
                ```

            Callback hooks are triggered at these points:

            - on_run_start: Before processing any tasks
            - on_task_start: Before processing a task (once per task, not per repeat)
            - on_task_repeat_start: Before each repetition of a task
            - on_task_repeat_end: After each repetition completes
            - on_task_end: After all repetitions of a task complete
            - on_run_end: After all tasks complete

            ```python
            # Typical usage
            benchmark = MyBenchmark(agent_data=config)
            reports = benchmark.run(tasks=tasks)

            # Analyze results
            for report in reports:
                print(f"Task {report['task_id']}, Repeat {report['repeat_idx']}: {report['eval']}")
                print(f"Config: {report['config']}")
                print(f"Traces: {report['traces']}")
            ```
        """
        # Normalize tasks into a TaskCollection
        if isinstance(tasks, Task):
            # Single task
            self.tasks = TaskCollection([tasks])
        elif isinstance(tasks, TaskCollection):
            self.tasks = tasks
        else:
            # Iterable of tasks or dicts
            self.tasks = TaskCollection.from_list(list(tasks))

        # Normalize agent_data into a list matching the number of tasks
        if isinstance(self.agent_data, dict):
            # Single config for all tasks
            agent_data_list: List[Dict[str, Any]] = [cast(Dict[str, Any], self.agent_data) for _ in range(len(self.tasks))]
        else:
            # Task-specific configs
            agent_data_list = list(self.agent_data)

        if len(agent_data_list) != len(self.tasks):
            raise ValueError(
                f"`agent_data` must either be a single dict or an iterable matching the number of tasks. "
                f"Got {len(agent_data_list)} agent configs for {len(self.tasks)} tasks."
            )

        # Clear reports from previous run() calls to prevent accumulation
        self.reports = []

        # Callbacks at the start of the run
        for cb in self.callbacks:
            cb.on_run_start(self)

        for task_idx, (task, agent_data) in enumerate(zip(self.tasks, agent_data_list)):
            # Callbacks at the start of each task
            for cb in self.callbacks:
                cb.on_task_start(self, task)

            for repeat_idx in range(self.n_task_repeats):
                for cb in self.callbacks:
                    cb.on_task_repeat_start(self, task, repeat_idx)

                # Initialize status and error tracking
                execution_status = TaskExecutionStatus.SUCCESS
                error_info: Optional[Dict[str, Any]] = None
                final_answers: Any = None
                eval_results: Any = None
                execution_traces: Dict[str, Any] = {}
                execution_configs: Dict[str, Any] = {}

                try:
                    # 1. Setup
                    environment = self.setup_environment(agent_data, task)
                    user = self.setup_user(agent_data, environment, task)
                    if user is None and self.max_invocations > 1:
                        # Warn if multi-turn is enabled but no user to drive interaction
                        warnings.warn(
                            f"max_invocations={self.max_invocations} > 1 but no user simulator provided. "
                            f"Falling back to single-turn execution for task {task.id}."
                        )
                    agents_to_run, agents_dict = self.setup_agents(agent_data, environment, task, user)
                    evaluators = self.setup_evaluators(environment, task, agents_to_run, user)

                    # Auto-register components returned from setup methods
                    # Environment
                    if environment is not None and isinstance(environment, TraceableMixin):
                        self.register("environment", "env", environment)

                    # User
                    if user is not None and isinstance(user, TraceableMixin):
                        self.register("user", "user", user)

                    # Agents (use their names from agents_dict)
                    for agent_name, agent in agents_dict.items():
                        if isinstance(agent, TraceableMixin):
                            self.register("agents", agent_name, agent)

                except Exception as e:
                    # Setup failed - record error and optionally re-raise
                    execution_status = TaskExecutionStatus.SETUP_FAILED
                    error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": "".join(__import__("traceback").format_exception(type(e), e, e.__traceback__)),
                    }

                    # Create a minimal report for this failed setup
                    # Use metadata task_id if available (e.g., tau2), else fall back to UUID
                    report = {
                        "task_id": task.metadata.get("task_id", str(task.id)),
                        "repeat_idx": repeat_idx,
                        "status": execution_status.value,
                        "error": error_info,
                        "traces": {},
                        "config": {},
                        "eval": None,
                    }
                    self.reports.append(report)

                    for cb in self.callbacks:
                        cb.on_task_repeat_end(self, report)

                    # Clear registry before potentially re-raising
                    self.clear_registry()

                    if self.fail_on_setup_error:
                        raise

                    # Continue to next task repetition
                    continue

                # 2. Execute agent system with optional user interaction loop
                try:
                    final_answers = self.execution_loop(agents_to_run, task, environment, user)
                except AgentError as e:
                    # Agent violated contract at boundary (agent's fault)
                    execution_status = TaskExecutionStatus.AGENT_ERROR
                    error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "component": e.component,
                        "details": e.details,
                        "traceback": "".join(__import__("traceback").format_exception(type(e), e, e.__traceback__)),
                    }

                    if self.fail_on_task_error:
                        # Clear registry before re-raising
                        self.clear_registry()
                        raise

                    # Continue with trace collection even if task failed
                    final_answers = None
                except EnvironmentError as e:
                    # Environment/tool infrastructure failed (not agent's fault)
                    execution_status = TaskExecutionStatus.ENVIRONMENT_ERROR
                    error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "component": e.component,
                        "details": e.details,
                        "traceback": "".join(__import__("traceback").format_exception(type(e), e, e.__traceback__)),
                    }

                    if self.fail_on_task_error:
                        # Clear registry before re-raising
                        self.clear_registry()
                        raise

                    # Continue with trace collection even if task failed
                    final_answers = None
                except UserError as e:
                    # User simulator failed (not agent's fault)
                    execution_status = TaskExecutionStatus.USER_ERROR
                    error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "component": e.component,
                        "details": e.details,
                        "traceback": "".join(__import__("traceback").format_exception(type(e), e, e.__traceback__)),
                    }

                    if self.fail_on_task_error:
                        # Clear registry before re-raising
                        self.clear_registry()
                        raise

                    # Continue with trace collection even if task failed
                    final_answers = None
                except Exception as e:
                    # Unclassified error (e.g., agent framework internal failure)
                    execution_status = TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR
                    error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": "".join(__import__("traceback").format_exception(type(e), e, e.__traceback__)),
                    }

                    if self.fail_on_task_error:
                        # Clear registry before re-raising
                        self.clear_registry()
                        raise

                    # Continue with trace collection even if task failed
                    final_answers = None

                # # Callbacks before evaluation
                # for cb in self.callbacks:
                #     cb.on_before_evaluation(self, task, agent_output)

                # 3. Collect traces and configs (always attempt this)
                try:
                    execution_configs = self.collect_all_configs()
                    execution_traces = self.collect_all_traces()
                except Exception as e:
                    # If trace/config collection fails, record it but continue
                    execution_configs = {
                        "error": f"Failed to collect configs: {e}",
                        "error_type": type(e).__name__,
                    }
                    execution_traces = {
                        "error": f"Failed to collect traces: {e}",
                        "error_type": type(e).__name__,
                    }

                # 4. Evaluate (skip if task execution failed, unless we want partial evaluation)
                if execution_status == TaskExecutionStatus.SUCCESS:
                    try:
                        eval_results = self.evaluate(evaluators, agents_dict, final_answers, execution_traces)
                    except Exception as e:
                        execution_status = TaskExecutionStatus.EVALUATION_FAILED
                        error_info = {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "traceback": "".join(__import__("traceback").format_exception(type(e), e, e.__traceback__)),
                        }

                        if self.fail_on_evaluation_error:
                            # Clear registry before re-raising
                            self.clear_registry()
                            raise

                        # Set eval_results to None on failure
                        eval_results = None
                else:
                    # Task execution failed, so skip evaluation
                    eval_results = None

                # 5. Store results with status and error info
                # Use metadata task_id if available (e.g., tau2), else fall back to UUID
                report = {
                    "task_id": task.metadata.get("task_id", str(task.id)),
                    "repeat_idx": repeat_idx,
                    "status": execution_status.value,
                    "traces": execution_traces,
                    "config": execution_configs,
                    "eval": eval_results,
                }

                # Add error info if present
                if error_info is not None:
                    report["error"] = error_info

                self.reports.append(report)

                for cb in self.callbacks:
                    cb.on_task_repeat_end(self, report)

                # Clear registry after task repetition completes
                self.clear_registry()

            # Callbacks at the end of each task
            # Pass the last report for this task to the callback
            current_task_id = task.metadata.get("task_id", str(task.id))
            task_reports = [r for r in self.reports if r["task_id"] == current_task_id]
            last_report = task_reports[-1] if task_reports else {}
            for cb in self.callbacks:
                cb.on_task_end(self, task, last_report)

        # Callbacks at the end of the run
        for cb in self.callbacks:
            cb.on_run_end(self, self.reports)
        return self.reports

    def get_failed_tasks(
        self,
        status_filter: Optional[Union[TaskExecutionStatus, List[TaskExecutionStatus]]] = None,
        reports: Optional[List[Dict[str, Any]]] = None,
    ) -> TaskCollection:
        """Get tasks that failed during benchmark execution.

        This method retrieves failed tasks based on their execution status, useful for
        debugging, retry logic, or failure analysis.

        Args:
            status_filter: Filter by specific failure status(es). If None, returns all failed tasks
                (any status except SUCCESS). Can be a single TaskExecutionStatus or a list of them.
                Examples:
                - TaskExecutionStatus.TASK_EXECUTION_FAILED: Only tasks that failed during execution
                - TaskExecutionStatus.EVALUATION_FAILED: Only tasks where evaluation failed
                - [TaskExecutionStatus.TASK_EXECUTION_FAILED, TaskExecutionStatus.SETUP_FAILED]:
                  Tasks that failed during execution or setup
            reports: Optional list of reports to analyze. If None, uses the reports from the last
                run() call. This allows analyzing externally stored or modified reports.

        Returns:
            TaskCollection containing the failed tasks. Empty if no failures match the filter.

        Raises:
            RuntimeError: If reports is None and run() has not been executed yet.

        How to use:
            ```python
            # Run benchmark
            benchmark = MyBenchmark(agent_data=config)
            reports = benchmark.run(tasks=tasks)

            # Get all failed tasks (from internal state)
            failed = benchmark.get_failed_tasks()
            print(f"Failed: {len(failed)}/{len(benchmark.tasks)} tasks")

            # Or work with returned reports (safe from internal state changes)
            failed = benchmark.get_failed_tasks(reports=reports)

            # Get only tasks that failed during execution (not evaluation)
            execution_failures = benchmark.get_failed_tasks(
                TaskExecutionStatus.TASK_EXECUTION_FAILED,
                reports=reports
            )

            # Get setup and execution failures
            critical_failures = benchmark.get_failed_tasks(
                status_filter=[
                    TaskExecutionStatus.SETUP_FAILED,
                    TaskExecutionStatus.TASK_EXECUTION_FAILED
                ],
                reports=reports
            )

            # Retry failed tasks elegantly - this is the key use case!
            if len(failed) > 0:
                retry_reports = benchmark.run(tasks=failed)

            # Or more concisely
            reports = benchmark.run(tasks=tasks)
            retry_reports = benchmark.run(tasks=benchmark.get_failed_tasks())
            ```
        """
        # Use provided reports or fall back to internal state
        if reports is None:
            if not self.reports:
                raise RuntimeError("get_failed_tasks() must be called after run(). No reports found.")
            reports = self.reports

        # Normalize status_filter to a set of status values (strings)
        if status_filter is None:
            # All non-success statuses (includes all classified and unclassified failures)
            filter_values = {
                TaskExecutionStatus.AGENT_ERROR.value,
                TaskExecutionStatus.ENVIRONMENT_ERROR.value,
                TaskExecutionStatus.USER_ERROR.value,
                TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR.value,
                TaskExecutionStatus.EVALUATION_FAILED.value,
                TaskExecutionStatus.SETUP_FAILED.value,
                # Include deprecated status for backward compatibility
                TaskExecutionStatus.TASK_EXECUTION_FAILED.value,
            }
        elif isinstance(status_filter, TaskExecutionStatus):
            filter_values = {status_filter.value}
        else:
            filter_values = {status.value for status in status_filter}

        # Collect unique failed task IDs matching the filter
        failed_task_ids = set()
        for report in reports:
            if report["status"] in filter_values:
                failed_task_ids.add(report["task_id"])

        # Build TaskCollection from original tasks that failed
        # Use metadata task_id if available (e.g., tau2), else fall back to UUID
        failed_tasks = [task for task in self.tasks if task.metadata.get("task_id", str(task.id)) in failed_task_ids]
        return TaskCollection(failed_tasks)
