"""Tests for MultiAgentBench benchmark classes."""


from maseval import Task
from maseval.benchmark.multiagentbench import (
    MultiAgentBenchEnvironment,
    MultiAgentBenchEvaluator,
)


class TestMultiAgentBenchBenchmark:
    """Tests for MultiAgentBenchBenchmark abstract class."""

    def test_setup_environment_returns_correct_type(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """setup_environment should return MultiAgentBenchEnvironment."""
        env = benchmark_instance.setup_environment({}, sample_research_task)

        assert isinstance(env, MultiAgentBenchEnvironment)

    def test_setup_user_returns_none(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """setup_user should return None (no user simulator for multi-agent)."""
        env = benchmark_instance.setup_environment({}, sample_research_task)
        user = benchmark_instance.setup_user({}, env, sample_research_task)

        assert user is None

    def test_setup_evaluators_returns_evaluator(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """setup_evaluators should return MultiAgentBenchEvaluator."""
        env = benchmark_instance.setup_environment({}, sample_research_task)
        agents, _ = benchmark_instance.setup_agents({}, env, sample_research_task, None)
        evaluators = benchmark_instance.setup_evaluators(env, sample_research_task, agents, None)

        assert len(evaluators) == 1
        assert isinstance(evaluators[0], MultiAgentBenchEvaluator)

    def test_setup_agents_creates_correct_count(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """setup_agents should create correct number of agents."""
        env = benchmark_instance.setup_environment({}, sample_research_task)
        agents_list, agents_dict = benchmark_instance.setup_agents({}, env, sample_research_task, None)

        # sample_research_task has 2 agents
        assert len(agents_list) == 2
        assert len(agents_dict) == 2
        assert "agent1" in agents_dict
        assert "agent2" in agents_dict

    def test_run_agents_executes_all_agents(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """run_agents should execute all agents and return results."""
        env = benchmark_instance.setup_environment({}, sample_research_task)
        agents_list, _ = benchmark_instance.setup_agents({}, env, sample_research_task, None)

        results = benchmark_instance.run_agents(
            agents_list,
            sample_research_task,
            env,
            sample_research_task.query,
        )

        assert isinstance(results, list)
        assert len(results) == 2
        assert all("agent_id" in r for r in results)
        assert all("result" in r for r in results)

    def test_evaluate_calls_evaluators(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """evaluate should call all evaluators."""
        env = benchmark_instance.setup_environment({}, sample_research_task)
        agents_list, agents_dict = benchmark_instance.setup_agents({}, env, sample_research_task, None)
        evaluators = benchmark_instance.setup_evaluators(env, sample_research_task, agents_list, None)

        final_answer = [{"agent_id": "agent1", "result": "Done"}]
        traces = {
            "agents": {"agent1": {"token_usage": 100, "action_log": [], "communication_log": []}},
            "environment": {},
        }

        results = benchmark_instance.evaluate(evaluators, agents_dict, final_answer, traces)

        assert len(results) == 1
        assert "passed" in results[0]


class TestBenchmarkIntegration:
    """Integration tests for benchmark execution."""

    def test_run_single_task(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """Benchmark should run a single task end-to-end."""
        results = benchmark_instance.run(
            tasks=[sample_research_task],
            agent_data={},
        )

        assert len(results) == 1
        assert results[0]["task_id"] == "research_1"
        assert "status" in results[0]
        assert "traces" in results[0]
        assert "eval" in results[0]

    def test_run_multiple_tasks(
        self,
        benchmark_instance,
        sample_research_task: Task,
        sample_bargaining_task: Task,
    ):
        """Benchmark should run multiple tasks."""
        results = benchmark_instance.run(
            tasks=[sample_research_task, sample_bargaining_task],
            agent_data={},
        )

        assert len(results) == 2

    def test_run_with_task_repeats(
        self,
        concrete_multiagentbench_benchmark,
        sample_research_task: Task,
    ):
        """Benchmark should repeat tasks when n_task_repeats > 1."""
        benchmark = concrete_multiagentbench_benchmark(
            n_task_repeats=3,
            progress_bar=False,
        )
        results = benchmark.run(
            tasks=[sample_research_task],
            agent_data={},
        )

        assert len(results) == 3
        assert all(r["task_id"] == "research_1" for r in results)
        assert [r["repeat_idx"] for r in results] == [0, 1, 2]

    def test_run_collects_traces(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """Benchmark should collect traces from all components."""
        results = benchmark_instance.run(
            tasks=[sample_research_task],
            agent_data={},
        )

        traces = results[0]["traces"]
        assert "agents" in traces
        assert "environment" in traces or traces.get("environment") is None

    def test_run_handles_setup_error(
        self,
        concrete_multiagentbench_benchmark,
        sample_research_task: Task,
    ):
        """Benchmark should handle setup errors gracefully."""

        class FailingBenchmark(concrete_multiagentbench_benchmark):
            def setup_environment(self, agent_data, task):
                raise RuntimeError("Setup failed")

        benchmark = FailingBenchmark(progress_bar=False)
        results = benchmark.run(
            tasks=[sample_research_task],
            agent_data={},
        )

        assert len(results) == 1
        assert results[0]["status"] == "setup_failed"
        assert "error" in results[0]


class TestAgentCreation:
    """Tests for agent creation and configuration."""

    def test_agents_have_correct_ids(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """Created agents should have IDs from task config."""
        env = benchmark_instance.setup_environment({}, sample_research_task)
        _, agents_dict = benchmark_instance.setup_agents({}, env, sample_research_task, None)

        assert "agent1" in agents_dict
        assert "agent2" in agents_dict

    def test_agents_have_profiles(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """Created agents should have profiles from task config."""
        env = benchmark_instance.setup_environment({}, sample_research_task)
        _, agents_dict = benchmark_instance.setup_agents({}, env, sample_research_task, None)

        agent1 = agents_dict["agent1"]
        assert hasattr(agent1, "profile")
        assert "machine learning" in agent1.profile.lower()


class TestEvaluatorConfiguration:
    """Tests for evaluator configuration."""

    def test_evaluator_uses_task_model_id(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """Evaluator should use model_id from task evaluation_data."""
        env = benchmark_instance.setup_environment({}, sample_research_task)
        agents, _ = benchmark_instance.setup_agents({}, env, sample_research_task, None)
        evaluators = benchmark_instance.setup_evaluators(env, sample_research_task, agents, None)

        evaluator = evaluators[0]
        assert evaluator.domain == "research"

    def test_evaluator_domain_from_task(
        self,
        benchmark_instance,
        sample_bargaining_task: Task,
    ):
        """Evaluator should get domain from task environment_data."""
        env = benchmark_instance.setup_environment({}, sample_bargaining_task)
        agents, _ = benchmark_instance.setup_agents({}, env, sample_bargaining_task, None)
        evaluators = benchmark_instance.setup_evaluators(env, sample_bargaining_task, agents, None)

        evaluator = evaluators[0]
        assert evaluator.domain == "bargaining"
