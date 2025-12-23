"""Tau 2 Benchmark - Main Implementation.

Framework-agnostic implementation of the tau2-bench benchmark for evaluating
LLM-based agents on customer service tasks across multiple domains.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Reference Paper: "Tau-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains"
https://arxiv.org/abs/2406.12045

Usage:
    from maseval.benchmark.tau2 import (
        Tau2Benchmark, Tau2Environment, Tau2Evaluator, Tau2User,
        load_tasks, configure_model_ids,
    )

    # Load data and configure model IDs
    tasks = load_tasks("retail", split="base", limit=5)
    configure_model_ids(
        tasks,
        user_model_id="gpt-4o",
        evaluator_model_id="gpt-4o",  # Optional - only for NL assertions
    )

    # Create your framework-specific benchmark subclass
    class MyTau2Benchmark(Tau2Benchmark):
        def setup_agents(self, agent_data, environment, task, user):
            # Your framework-specific agent creation
            ...

        def get_model_adapter(self, model_id, **kwargs):
            # Create and optionally register model adapters
            adapter = MyModelAdapter(model_id)
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    # Run
    benchmark = MyTau2Benchmark(agent_data={})
    results = benchmark.run(tasks)
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

from maseval import AgentAdapter, Benchmark, Evaluator, ModelAdapter, Task, User
from maseval.core.agentic_user import AgenticUser

from maseval.benchmark.tau2.environment import Tau2Environment
from maseval.benchmark.tau2.evaluator import Tau2Evaluator


# =============================================================================
# User Simulator
# =============================================================================


class Tau2User(AgenticUser):
    """Tau2-specific user simulator with customer service personas.

    Extends the AgenticUser class with tau2-specific behavior:
    - Customer personas from user_scenario
    - Domain-aware responses (airline, retail, telecom)
    - Multi-turn interaction support
    - Tool usage capabilities

    Note: This is a base class. Framework-specific subclasses should override
    get_tool() to return a compatible tool.
    """

    DEFAULT_MAX_TURNS = 10  # Higher than MACS due to more complex tasks
    DEFAULT_STOP_TOKEN = "</stop>"
    DEFAULT_EARLY_STOPPING_CONDITION = "The user's issue has been fully resolved by the agent"

    def __init__(
        self,
        model: ModelAdapter,
        scenario: str,
        initial_query: str,
        name: str = "Customer",
        template: Optional[str] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        stop_token: str = DEFAULT_STOP_TOKEN,
        early_stopping_condition: str = DEFAULT_EARLY_STOPPING_CONDITION,
        tools: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize Tau2 user simulator.

        Args:
            model: ModelAdapter for LLM-based response generation
            scenario: Full scenario text containing user instructions
            initial_query: The initial query to the agent
            name: User name for identification (default: "Customer")
            template: Optional custom prompt template
            max_turns: Maximum conversation turns
            stop_token: Token indicating user satisfaction
            early_stopping_condition: Description of when to emit stop token
            tools: Optional dictionary of tools available to the user
        """
        # Extract user profile from scenario
        user_profile = self._extract_user_profile(scenario)

        super().__init__(
            name=name,
            model=model,
            user_profile=user_profile,
            scenario=scenario,
            initial_query=initial_query,
            template=template,
            max_turns=max_turns,
            stop_token=stop_token,
            early_stopping_condition=early_stopping_condition,
            tools=tools,
        )

    def get_tool(self) -> Any:
        """Return a tool for agent interaction.

        This base implementation raises NotImplementedError.
        Framework-specific subclasses should override this method.

        Raises:
            NotImplementedError: Always, as this must be implemented by subclass.
        """
        raise NotImplementedError("Tau2User.get_tool() must be overridden by framework-specific subclass.")

    @staticmethod
    def _extract_user_profile(scenario: str) -> Dict[str, Any]:
        """Extract user profile from scenario text.

        Args:
            scenario: Full scenario/instructions text

        Returns:
            Dict with user profile fields
        """
        profile: Dict[str, Any] = {}

        # Parse structured format if present
        if "Persona:" in scenario or "persona:" in scenario:
            # Try to extract persona section
            for prefix in ["Persona:", "persona:"]:
                if prefix in scenario:
                    parts = scenario.split(prefix, 1)
                    if len(parts) > 1:
                        persona_section = parts[1].split("\n")[0].strip()
                        profile["persona"] = persona_section

        # Include full scenario as context
        profile["full_scenario"] = scenario

        return profile

    def gather_traces(self) -> Dict[str, Any]:
        """Gather traces with Tau2-specific information."""
        traces = super().gather_traces()
        traces.update(
            {
                "max_turns": self.max_turns,
                "turns_used": self._turn_count,
                "stopped_by_user": self._stopped,
            }
        )
        return traces


# =============================================================================
# Benchmark
# =============================================================================


class Tau2Benchmark(Benchmark):
    """Tau2 Benchmark - Framework-agnostic base class.

    This base class handles:
    - Environment setup with Tau2Environment (real tools)
    - Deterministic evaluation via database state comparison
    - Optional user simulation for multi-turn tasks

    Users must subclass and implement:
    - setup_agents() for their agent framework
    - get_model_adapter() to provide model adapters

    Model IDs for components are read from task data:
    - task.user_data["model_id"] for user simulator
    - task.evaluation_data["model_id"] for NL assertion evaluator (optional)

    Use configure_model_ids() to set these values after loading tasks.

    Example:
        class MyTau2Benchmark(Tau2Benchmark):
            def setup_agents(self, agent_data, environment, task, user):
                # Setup your agents here
                ...

            def get_model_adapter(self, model_id, **kwargs):
                return MyModelAdapter(model_id)

        tasks = load_tasks("retail")
        configure_model_ids(tasks, user_model_id="gpt-4o")

        benchmark = MyTau2Benchmark(agent_data={})
        benchmark.run(tasks)
    """

    def __init__(
        self,
        agent_data: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
        n_task_repeats: int = 1,
        max_invocations: int = 10,
        data_dir: Optional[Path] = None,
        **kwargs: Any,
    ):
        """Initialize benchmark.

        Args:
            agent_data: Agent configuration dict
            callbacks: Benchmark callbacks
            n_task_repeats: Repetitions per task (for Pass@k metrics, use k here)
            max_invocations: Maximum agent-user interaction rounds
            data_dir: Base data directory for domain data
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(agent_data or {}, callbacks, n_task_repeats, max_invocations, **kwargs)
        self._data_dir = data_dir

    def _get_user_model_id(self, task: Task) -> str:
        """Get user simulator model ID from task.user_data.

        Raises:
            ValueError: If model_id not configured in task.user_data
        """
        model_id = task.user_data.get("model_id")
        if model_id is None:
            raise ValueError(
                "User simulator model_id not configured in task.user_data.\n"
                "Use configure_model_ids() after loading tasks:\n\n"
                "    from maseval.benchmark.tau2 import load_tasks, configure_model_ids\n\n"
                "    tasks = load_tasks('retail')\n"
                "    configure_model_ids(\n"
                "        tasks,\n"
                "        user_model_id='gpt-4o',\n"
                "    )"
            )
        return model_id

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
    ) -> Tau2Environment:
        """Create environment for a task.

        Creates a Tau2Environment with real tool implementations
        for the task's domain.

        Args:
            agent_data: Agent configuration
            task: Current task

        Returns:
            Tau2Environment instance
        """
        return Tau2Environment(
            task_data=task.environment_data,
            data_dir=self._data_dir,
        )

    def setup_user(  # type: ignore[override]
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
    ) -> Optional[User]:
        """Create Tau2 user simulator.

        Creates a Tau2User with scenario from the task.
        Model ID is read from task.user_data["model_id"].

        Note: Tau2User.get_tool() raises NotImplementedError.
        Framework-specific subclasses should wrap this user
        or override setup_user() to return a user with get_tool() implemented.

        Args:
            agent_data: Agent configuration
            environment: The task environment
            task: Current task with user scenario

        Returns:
            Tau2User instance
        """
        # Build scenario from user instructions
        user_data = task.user_data
        instructions = user_data.get("instructions", {})

        if isinstance(instructions, str):
            scenario = instructions
        elif isinstance(instructions, dict):
            parts = []
            if instructions.get("reason_for_call"):
                parts.append(f"Reason for call: {instructions['reason_for_call']}")
            if instructions.get("known_info"):
                parts.append(f"Known info: {instructions['known_info']}")
            if instructions.get("task_instructions"):
                parts.append(f"Task: {instructions['task_instructions']}")
            scenario = "\n".join(parts)
        else:
            scenario = ""

        # Add persona if available
        persona = user_data.get("persona")
        if persona:
            scenario = f"Persona: {persona}\n\n{scenario}"

        user_model_id = self._get_user_model_id(task)

        # Get user tools from environment
        user_tools = environment.create_user_tools()

        return Tau2User(
            model=self.get_model_adapter(
                user_model_id,
                register_name="user_simulator",
            ),
            scenario=scenario,
            initial_query=task.query,
            tools=user_tools,
        )

    @abstractmethod
    def setup_agents(  # type: ignore[override]
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create agents for this task. Must be implemented by subclass.

        Args:
            agent_data: Agent configuration
            environment: Tau2Environment with real tools
            task: Current task
            user: Optional user simulator

        Returns:
            Tuple of (ordered agent list, agent dict keyed by ID)
        """
        pass

    def setup_evaluators(  # type: ignore[override]
        self,
        environment: Tau2Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
    ) -> Sequence[Evaluator]:
        """Create evaluator for the task.

        Creates a Tau2Evaluator that performs deterministic
        database state comparison.

        Args:
            environment: Tau2Environment instance
            task: Current task with evaluation criteria
            agents: Agent instances
            user: Optional user simulator

        Returns:
            List with single Tau2Evaluator instance
        """
        return [
            Tau2Evaluator(
                task=task,
                environment=environment,
                data_dir=self._data_dir,
            )
        ]

    def run_agents(  # type: ignore[override]
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Tau2Environment,
        query: str = "",
    ) -> Any:
        """Execute agents and return final answer.

        Args:
            agents: Agent instances to run
            task: Current task
            environment: Tau2Environment
            query: Query/prompt for agents

        Returns:
            Final answer from agents
        """
        answers = [agent.run(query) for agent in agents]
        return answers[0] if len(answers) == 1 else answers

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate using Tau2 evaluators.

        Uses each evaluator's filter_traces() method to extract relevant data,
        then calls the evaluator with the filtered traces.

        Returns tau2 format:
        - reward: Float [0.0, 1.0]
        - passed: Boolean
        - reward_breakdown: Per-evaluator scores
        - env_check, action_check, communicate_check: Detailed results

        Args:
            evaluators: List of evaluators
            agents: Dict of agents
            final_answer: Final answer from agents
            traces: Execution traces

        Returns:
            List of evaluation result dicts
        """
        results = []
        for evaluator in evaluators:
            filtered_traces = evaluator.filter_traces(traces)
            result = evaluator(filtered_traces, final_answer)
            results.append(result)

        return results
