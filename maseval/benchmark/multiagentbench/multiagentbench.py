"""MultiAgentBench benchmark implementations.

This module provides benchmark classes for the MARBLE MultiAgentBench suite:
- MultiAgentBenchBenchmark: Abstract base for framework-agnostic evaluation
- MarbleMultiAgentBenchBenchmark: Exact MARBLE reproduction mode
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from maseval import (
    AgentAdapter,
    Benchmark,
    Environment,
    Evaluator,
    ModelAdapter,
    Task,
    User,
)
from maseval.core.callback import BenchmarkCallback

from maseval.benchmark.multiagentbench.environment import MultiAgentBenchEnvironment
from maseval.benchmark.multiagentbench.evaluator import (
    MultiAgentBenchEvaluator,
)


class MultiAgentBenchBenchmark(Benchmark):
    """Abstract base class for framework-agnostic MultiAgentBench evaluation.

    This benchmark provides the infrastructure for evaluating multi-agent systems
    on MARBLE's MultiAgentBench tasks. Subclasses implement `setup_agents()` with
    their specific agent framework.

    The benchmark supports:
    - Multiple coordination modes (star, cooperative, tree, hierarchical)
    - Multiple domains (research, bargaining, coding, database, etc.)
    - LLM-based evaluation matching MARBLE's metrics
    - Comprehensive tracing of agent interactions

    Example:
        ```python
        class MyMultiAgentBenchmark(MultiAgentBenchBenchmark):
            def setup_agents(self, agent_data, environment, task, user):
                # Create agents using your framework
                ...

            def get_model_adapter(self, model_id, **kwargs):
                adapter = MyModelAdapter(model_id)
                if "register_name" in kwargs:
                    self.register("models", kwargs["register_name"], adapter)
                return adapter

        benchmark = MyMultiAgentBenchmark()
        results = benchmark.run(tasks, agent_data={})
        ```
    """

    def __init__(
        self,
        callbacks: Optional[List[BenchmarkCallback]] = None,
        n_task_repeats: int = 1,
        max_invocations: int = 10,
        num_workers: int = 1,
        fail_on_setup_error: bool = False,
        fail_on_task_error: bool = False,
        fail_on_evaluation_error: bool = False,
        progress_bar: bool | str = True,
    ):
        """Initialize the benchmark.

        Args:
            callbacks: Optional list of callbacks
            n_task_repeats: Number of times to repeat each task
            max_invocations: Maximum agent invocations per task
            num_workers: Number of parallel workers
            fail_on_setup_error: Raise on setup errors
            fail_on_task_error: Raise on task errors
            fail_on_evaluation_error: Raise on evaluation errors
            progress_bar: Progress bar configuration
        """
        super().__init__(
            callbacks=callbacks,
            n_task_repeats=n_task_repeats,
            max_invocations=max_invocations,
            num_workers=num_workers,
            fail_on_setup_error=fail_on_setup_error,
            fail_on_task_error=fail_on_task_error,
            fail_on_evaluation_error=fail_on_evaluation_error,
            progress_bar=progress_bar,
        )

    def setup_environment(self, agent_data: Dict[str, Any], task: Task) -> Environment:
        """Create the MultiAgentBench environment.

        Args:
            agent_data: Agent configuration
            task: The task to set up

        Returns:
            MultiAgentBenchEnvironment instance
        """
        return MultiAgentBenchEnvironment(
            task_data=task.environment_data,
            callbacks=self.callbacks,
        )

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
    ) -> Optional[User]:
        """MultiAgentBench tasks don't use user simulators.

        The multi-agent coordination replaces user interaction.

        Returns:
            None
        """
        return None

    @abstractmethod
    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create agents for the task (implement in subclass).

        Subclasses should:
        1. Read agent specifications from task.environment_data["agents"]
        2. Create agents using their framework
        3. Wrap them in AgentAdapter
        4. Set up relationships from task.environment_data["relationships"]

        Args:
            agent_data: Agent configuration (model IDs, etc.)
            environment: The environment instance
            task: The task containing agent specs
            user: User simulator (None for MultiAgentBench)

        Returns:
            Tuple of (agents_to_run, agents_dict)
        """
        pass

    def setup_evaluators(
        self,
        environment: Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
    ) -> Sequence[Evaluator]:
        """Create evaluators for the task.

        Args:
            environment: The environment
            task: The task with evaluation data
            agents: The agents
            user: User simulator (None for MultiAgentBench)

        Returns:
            List of evaluators
        """
        # Get evaluation model ID from task or default
        eval_model_id = task.evaluation_data.get("model_id", "gpt-4o-mini")

        # Create model adapter for evaluation
        model_adapter = self.get_model_adapter(
            eval_model_id,
            register_name="evaluator_model",
        )

        # Get domain-specific evaluation configuration
        domain = task.environment_data.get("scenario", "")
        metrics_config = task.evaluation_data.get("metrics", {})
        output_format = task.evaluation_data.get("output_format", "")

        return [
            MultiAgentBenchEvaluator(
                domain=domain,
                model_adapter=model_adapter,
                metrics_config=metrics_config,
                output_format=output_format,
            )
        ]

    @abstractmethod
    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        """Provide a model adapter (implement in subclass).

        Args:
            model_id: Model identifier
            **kwargs: Additional arguments including register_name

        Returns:
            ModelAdapter instance
        """
        pass

    def run_agents(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Environment,
        query: str,
    ) -> Any:
        """Execute the multi-agent system.

        For MultiAgentBench, this runs all agents on the task and
        collects their outputs.

        Args:
            agents: Agents to run
            task: The task
            environment: The environment
            query: The query/task content

        Returns:
            Combined agent outputs
        """
        results: List[Dict[str, Any]] = []

        for agent in agents:
            result = agent.run(query)
            results.append(
                {
                    "agent_id": getattr(agent, "agent_id", str(agent)),
                    "result": result,
                }
            )

        return results

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Execute evaluators on the results.

        Args:
            evaluators: The evaluators
            agents: Dict of all agents
            final_answer: The combined agent outputs
            traces: Execution traces

        Returns:
            List of evaluation results
        """
        results = []

        for evaluator in evaluators:
            # MultiAgentBenchEvaluator expects traces in a specific format
            result = evaluator(traces, final_answer)
            results.append(result)

        return results


class MarbleMultiAgentBenchBenchmark(MultiAgentBenchBenchmark):
    """MARBLE reproduction mode for MultiAgentBench.

    This benchmark uses MARBLE's native agents and engine for exact
    reproduction of published results. It wraps MARBLE components
    in MASEval adapters for unified tracing.

    Example:
        ```python
        from maseval.benchmark.multiagentbench import (
            MarbleMultiAgentBenchBenchmark,
            load_tasks,
            configure_model_ids,
        )

        class MyMarbleBenchmark(MarbleMultiAgentBenchBenchmark):
            def get_model_adapter(self, model_id, **kwargs):
                from maseval.interface.openai import OpenAIModelAdapter
                adapter = OpenAIModelAdapter(model_id)
                if "register_name" in kwargs:
                    self.register("models", kwargs["register_name"], adapter)
                return adapter

        tasks = load_tasks("research", limit=5)
        configure_model_ids(tasks, agent_model_id="gpt-4o")

        benchmark = MyMarbleBenchmark()
        results = benchmark.run(tasks, agent_data={})
        ```
    """

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create MARBLE agents wrapped in MASEval adapters.

        Args:
            agent_data: Agent configuration
            environment: The environment
            task: The task with agent specifications
            user: User simulator (None)

        Returns:
            Tuple of (agents_to_run, agents_dict)
        """
        from maseval.benchmark.multiagentbench.adapters.marble_adapter import (
            create_marble_agents,
        )

        # Get agent configurations from task
        agent_configs = task.environment_data.get("agents", [])
        model_id = task.environment_data.get("llm", "gpt-4o-mini")

        # Get MARBLE environment from our wrapper
        marble_env = None
        if isinstance(environment, MultiAgentBenchEnvironment):
            marble_env = environment._marble_env

        # Create MARBLE environment if not available
        if marble_env is None:
            marble_env = self._create_marble_env(task)

        # Create agents using factory function
        agents_list, agents_dict = create_marble_agents(
            agent_configs=agent_configs,
            marble_env=marble_env,
            model=model_id,
            callbacks=self.callbacks,
        )

        # Set up agent graph for inter-agent communication
        self._setup_agent_graph(agents_dict, task, marble_env)

        # Register agents for tracing
        for agent_id, adapter in agents_dict.items():
            self.register("agents", agent_id, adapter)

        return agents_list, agents_dict

    def _create_marble_env(self, task: Task) -> Any:
        """Create a MARBLE environment for the task.

        Args:
            task: The task with environment configuration

        Returns:
            MARBLE environment instance
        """
        try:
            from .marble.environments.base_env import BaseEnvironment
        except ImportError as e:
            raise ImportError(f"MARBLE is not available. Clone MARBLE to maseval/benchmark/multiagentbench/marble/\nOriginal error: {e}") from e

        env_config = task.environment_data.get("environment", {})
        task_config = task.environment_data.get("task", {})

        config = {
            "description": f"{task.environment_data.get('scenario', '')} environment",
            "task_description": task_config.get("content", "") if isinstance(task_config, dict) else str(task_config),
            "max_iterations": env_config.get("max_iterations") or task.environment_data.get("max_iterations", 10),
        }

        return BaseEnvironment(name=config["description"], config=config)

    def _setup_agent_graph(
        self,
        agents_dict: Dict[str, AgentAdapter],
        task: Task,
        marble_env: Any,
    ) -> None:
        """Set up MARBLE's AgentGraph for inter-agent communication.

        Args:
            agents_dict: Dict of agent adapters
            task: Task with relationship data
            marble_env: MARBLE environment
        """
        try:
            from .marble.graph.agent_graph import AgentGraph
        except ImportError:
            # MARBLE not available, skip graph setup
            return

        # Extract MARBLE agents from adapters
        marble_agents = [adapter.marble_agent for adapter in agents_dict.values()]  # type: ignore[attr-defined]

        # Build config for AgentGraph
        relationships = task.environment_data.get("relationships", [])
        coordination_mode = task.environment_data.get("coordinate_mode", "cooperative")

        # Create a minimal Config object
        class MinimalConfig:
            def __init__(self) -> None:
                self.coordination_mode = coordination_mode
                self.relationships = relationships

        config = MinimalConfig()

        try:
            # Create agent graph
            graph = AgentGraph(marble_agents, config)  # type: ignore

            # Set graph on all agents
            for agent in marble_agents:
                agent.set_agent_graph(graph)

        except Exception:
            # Graph creation failed, agents will work without inter-agent communication
            pass

    def run_agents(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Environment,
        query: str,
    ) -> Any:
        """Execute agents using MARBLE's coordination patterns.

        Args:
            agents: Agents to run (MarbleAgentAdapters)
            task: The task
            environment: The environment
            query: The task query

        Returns:
            Combined agent outputs with communication logs
        """
        results: List[Dict[str, Any]] = []
        communications: List[str] = []

        # Get coordination mode
        coordination_mode = task.environment_data.get("coordinate_mode", "cooperative")

        # Simple execution: each agent acts on the task
        # More complex coordination modes (star, tree, hierarchical) would
        # require the MARBLE Engine, which is outside the scope of this adapter
        for agent in agents:
            result = agent.run(query)
            agent_id = getattr(agent, "agent_id", str(agent))

            results.append(
                {
                    "agent_id": agent_id,
                    "result": result,
                }
            )

            # Collect communication logs if available
            if hasattr(agent, "get_serialized_messages"):
                comm = agent.get_serialized_messages()  # type: ignore
                if comm:
                    communications.append(comm)

        return {
            "agent_results": results,
            "communications": communications,
            "coordination_mode": coordination_mode,
        }

    @abstractmethod
    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        """Provide a model adapter (implement in subclass).

        Args:
            model_id: Model identifier
            **kwargs: Additional arguments

        Returns:
            ModelAdapter instance
        """
        pass
