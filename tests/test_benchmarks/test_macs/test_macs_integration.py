"""Integration tests for MACS benchmark components."""

import json
import pytest
from unittest.mock import patch

from maseval import Task
from maseval.benchmark.macs import (
    MACSEnvironment,
    MACSEvaluator,
    MACSUser,
    compute_benchmark_metrics,
)

from .conftest import MACSModelAdapter, ConcreteMACSBenchmark


# =============================================================================
# Environment and Tool Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestEnvironmentToolIntegration:
    """Integration tests for MACSEnvironment and MACSGenericTool."""

    def test_environment_creates_callable_tools(self, travel_task):
        """Environment creates tools that can be called."""
        # Use a model that returns valid JSON responses (ToolLLMSimulator expects {"text": ..., "details": ...})
        model = MACSModelAdapter(responses=['{"text": "Found flights: AA123, UA456", "details": {}}'])
        env = MACSEnvironment(
            task_data={"environment_data": travel_task.environment_data},
            model=model,
        )

        assert "search_flights" in env.tools
        assert "book_flight" in env.tools

        # Tools should be callable
        search_flights = env.tools["search_flights"]
        result = search_flights(origin="SFO", destination="JFK", date="2024-12-09")

        # Should return the text from the response
        assert "Found flights" in result

    def test_tool_tracks_invocations(self, travel_task):
        """Tool invocations are tracked in history."""
        model = MACSModelAdapter(responses=['{"text": "success", "details": {}}'])
        env = MACSEnvironment(
            task_data={"environment_data": travel_task.environment_data},
            model=model,
        )

        search_flights = env.tools["search_flights"]

        # Make multiple calls
        search_flights(origin="SFO", destination="JFK", date="2024-12-09")
        search_flights(origin="LAX", destination="ORD", date="2024-12-10")

        history = search_flights.history.to_list()
        assert len(history) == 2
        assert history[0]["inputs"]["origin"] == "SFO"
        assert history[1]["inputs"]["origin"] == "LAX"

    def test_agent_gets_subset_of_tools(self, macs_model):
        """Agent only gets tools from its assigned tool groups."""
        task_data = {
            "environment_data": {
                "tools": [
                    {
                        "tool_name": "group_a",
                        "actions": [{"name": "tool_a", "description": "Tool A"}],
                    },
                    {
                        "tool_name": "group_b",
                        "actions": [{"name": "tool_b", "description": "Tool B"}],
                    },
                ]
            }
        }
        env = MACSEnvironment(task_data, macs_model)

        agent_spec = {"agent_id": "agent", "tools": ["group_a"]}
        agent_tools = env.get_tools_for_agent(agent_spec)

        assert "tool_a" in agent_tools
        assert "tool_b" not in agent_tools


# =============================================================================
# Evaluator Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestEvaluatorIntegration:
    """Integration tests for MACSEvaluator."""

    def test_user_evaluation_with_conversation(self, travel_task, sample_conversation):
        """User evaluator works with real conversation trace."""
        response = json.dumps(
            [
                {
                    "assertion": "The user's flight booking request was acknowledged",
                    "answer": "TRUE",
                    "evidence": "Agent acknowledged the request",
                },
                {"assertion": "The user received flight options or a confirmation", "answer": "TRUE", "evidence": "Confirmation DL123456"},
            ]
        )
        model = MACSModelAdapter(responses=[response])
        evaluator = MACSEvaluator(model, travel_task, gsr_type="user")

        traces = {"messages": sample_conversation}
        result = evaluator(traces)

        assert result["gsr"] == 1.0
        assert len(result["report"]) == 2
        assert all(item["assertion_type"] == "user" for item in result["report"])

    def test_system_evaluation_with_tool_traces(self, travel_task, sample_conversation):
        """System evaluator includes tool invocations."""
        response = json.dumps(
            [
                {
                    "assertion": "The search_flights tool was called with correct parameters",
                    "answer": "TRUE",
                    "evidence": "Tool called with SFO, JFK",
                },
            ]
        )
        model = MACSModelAdapter(responses=[response])
        evaluator = MACSEvaluator(model, travel_task, gsr_type="system")

        tool_traces = {
            "search_flights": {
                "invocations": [
                    {"inputs": {"origin": "SFO", "destination": "JFK", "date": "2024-12-09"}, "outputs": "Found 3 flights", "status": "success"}
                ]
            }
        }

        traces = {"messages": sample_conversation, "tool_traces": tool_traces}
        result = evaluator(traces)

        assert result["gsr"] == 1.0
        assert result["report"][0]["assertion_type"] == "system"

        # Verify tool info was in the prompt
        prompt = model.prompts[0]
        assert "search_flights" in prompt
        assert "SFO" in prompt

    def test_evaluator_handles_partial_success(self, travel_task, sample_conversation):
        """Evaluator correctly computes partial GSR."""
        response = json.dumps(
            [
                {"assertion": "First assertion", "answer": "TRUE", "evidence": "OK"},
                {"assertion": "Second assertion", "answer": "FALSE", "evidence": "Failed"},
            ]
        )
        model = MACSModelAdapter(responses=[response])
        evaluator = MACSEvaluator(model, travel_task, gsr_type="user")

        traces = {"messages": sample_conversation}
        result = evaluator(traces)

        assert result["gsr"] == 0.0  # Not all passed
        assert result["partial_gsr"] == 0.5  # 1 of 2 passed


# =============================================================================
# User Simulator Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestUserSimulatorIntegration:
    """Integration tests for MACSUser."""

    def test_user_extracts_profile_from_scenario(self, macs_model, travel_task):
        """User correctly extracts profile from scenario."""
        user = MACSUser(
            model=macs_model,
            scenario=travel_task.metadata["scenario"],
            initial_prompt=travel_task.query,
        )

        # Should have extracted name and other details
        assert "full_scenario" in user.user_profile
        assert "Alice Johnson" in user.user_profile.get("full_scenario", "")

    def test_user_respects_max_turns(self, travel_task):
        """User simulator stops after max_turns."""
        model = MACSModelAdapter(responses=['{"text": "Yes", "details": {}}'] * 10)
        user = MACSUser(
            model=model,
            scenario=travel_task.metadata["scenario"],
            initial_prompt=travel_task.query,
            max_turns=3,
        )

        # Simulate turns
        for i in range(3):
            assert not user.is_done
            with patch.object(user.__class__.__bases__[0], "simulate_response", return_value="Response"):
                user.simulate_response(f"Question {i}")

        assert user.is_done
        assert user._turn_count == 3

    def test_user_detects_stop_token(self, travel_task):
        """User correctly detects and handles </stop> token."""
        model = MACSModelAdapter()
        user = MACSUser(
            model=model,
            scenario=travel_task.metadata["scenario"],
            initial_prompt=travel_task.query,
        )

        with patch.object(user.__class__.__bases__[0], "simulate_response", return_value="Great, that's all! </stop>"):
            response = user.simulate_response("Your flight is booked!")

        assert "</stop>" not in response
        assert user.is_done
        assert user._stopped


# =============================================================================
# Full Benchmark Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestFullBenchmarkIntegration:
    """End-to-end integration tests for MACS benchmark."""

    def test_complete_task_lifecycle(self, sample_agent_data, travel_task):
        """Test complete task: setup → run → evaluate."""
        # Model responses for various stages
        responses = [
            # Tool simulation (if tools are called)
            '{"flights": [{"id": "DL123", "time": "8:15am"}]}',
            # User evaluation
            json.dumps([{"assertion": "User acknowledged", "answer": "TRUE", "evidence": "OK"}]),
            # System evaluation
            json.dumps([{"assertion": "Tool called", "answer": "TRUE", "evidence": "OK"}]),
        ]
        model = MACSModelAdapter(responses=responses)
        benchmark = ConcreteMACSBenchmark(sample_agent_data, model)

        # Setup phase
        env = benchmark.setup_environment(sample_agent_data, travel_task)
        user = benchmark.setup_user(sample_agent_data, env, travel_task)
        agents_list, agents_dict = benchmark.setup_agents(sample_agent_data, env, travel_task, user)
        evaluators = benchmark.setup_evaluators(env, travel_task, agents_list, user)

        # Verify setup
        assert isinstance(env, MACSEnvironment)
        assert isinstance(user, MACSUser)
        assert len(agents_list) == 1
        assert len(evaluators) == 2

        # Run phase
        final_answer = benchmark.run_agents(agents_list, travel_task, env)
        assert final_answer is not None

        # Evaluate phase
        traces = {
            "agents": {
                "integration_agent": {
                    "messages": [
                        {"role": "user", "content": travel_task.query},
                        {"role": "assistant", "content": final_answer},
                    ]
                }
            },
            "tools": {},
        }
        results = benchmark.evaluate(evaluators, agents_dict, final_answer, traces)

        # Verify results
        assert len(results) == 1
        assert "user_gsr" in results[0]
        assert "system_gsr" in results[0]
        assert "overall_gsr" in results[0]

    def test_benchmark_aggregates_metrics_correctly(self):
        """Test metric aggregation across multiple tasks."""
        results = [
            {
                "task_id": "task-1",
                "eval": [{"overall_gsr": 1.0, "user_gsr": 1.0, "system_gsr": 1.0, "partial_gsr": 1.0}],
            },
            {
                "task_id": "task-2",
                "eval": [{"overall_gsr": 0.0, "user_gsr": 1.0, "system_gsr": 0.0, "partial_gsr": 0.5}],
            },
            {
                "task_id": "task-3",
                "eval": [{"overall_gsr": 1.0, "user_gsr": 1.0, "system_gsr": 1.0, "partial_gsr": 1.0}],
            },
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 3
        assert metrics["successful_tasks"] == 2  # overall_gsr == 1.0
        assert metrics["success_rate"] == pytest.approx(2 / 3)
        assert metrics["mean_metrics"]["overall_gsr"] == pytest.approx(2 / 3)
        assert metrics["mean_metrics"]["user_gsr"] == 1.0  # All user tests passed


# =============================================================================
# Data Loading Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestDataLoadingIntegration:
    """Integration tests for data loading with benchmark components."""

    def test_loaded_task_works_with_environment(self, macs_model, sample_agent_data):
        """Tasks loaded from data work with MACSEnvironment."""
        # Create a mock task that simulates loaded data
        task = Task(
            query="Book a flight",
            environment_data={
                "tools": [
                    {
                        "tool_name": "flight_search",
                        "actions": [{"name": "search", "description": "Search flights"}],
                    }
                ]
            },
            evaluation_data={"assertions": ["user: Booking done"]},
            metadata={"scenario": "Travel booking scenario", "task_id": "task-000001"},
        )

        benchmark = ConcreteMACSBenchmark(sample_agent_data, macs_model)
        env = benchmark.setup_environment(sample_agent_data, task)

        assert "search" in env.tools

    def test_loaded_agent_config_works_with_environment(self, macs_model):
        """Agent config works with tool assignment."""
        # Simulate loaded agent config
        agent_config = {
            "agents": [
                {"agent_id": "main", "agent_name": "Main Agent", "tools": ["tool_group"]},
            ],
            "primary_agent_id": "main",
        }

        task_data = {
            "environment_data": {
                "tools": [
                    {
                        "tool_name": "tool_group",
                        "actions": [{"name": "action1", "description": "Action 1"}],
                    }
                ]
            }
        }

        env = MACSEnvironment(task_data, macs_model)

        # Get tools for agent from config
        agent_spec = agent_config["agents"][0]
        agent_tools = env.get_tools_for_agent(agent_spec)

        assert "action1" in agent_tools


# =============================================================================
# Error Handling Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_evaluator_handles_malformed_llm_response(self, travel_task, sample_conversation):
        """Evaluator gracefully handles malformed LLM responses."""
        model = MACSModelAdapter(responses=["This is not valid JSON at all"])
        evaluator = MACSEvaluator(model, travel_task, gsr_type="user")

        traces = {"messages": sample_conversation}
        result = evaluator(traces)

        # Should return error result, not crash
        assert result["gsr"] == 0.0
        assert "error" in result

    def test_environment_handles_empty_tool_specs(self, macs_model):
        """Environment handles tasks with no tools."""
        task_data = {"environment_data": {"tools": []}}
        env = MACSEnvironment(task_data, macs_model)

        assert env.tools == {}

    def test_user_handles_missing_background(self, macs_model):
        """User simulator handles scenario without Background section."""
        scenario = "Simple goal: Book a hotel."

        user = MACSUser(
            model=macs_model,
            scenario=scenario,
            initial_prompt="Book a hotel",
        )

        # Should not crash, should have minimal profile
        assert "full_scenario" in user.user_profile


# =============================================================================
# End-to-End Pipeline Tests (benchmark.run())
# =============================================================================


@pytest.mark.benchmark
class TestEndToEndPipeline:
    """End-to-end tests that call benchmark.run() with TaskCollection.

    These tests verify the complete MACS benchmark pipeline:
    1. Task setup (environment, user, agents, evaluators)
    2. Agent execution
    3. Trace collection
    4. Evaluation (user GSR and system GSR)
    5. Report generation
    """

    def test_run_single_task(self, sample_agent_data, sample_task):
        """Run benchmark with a single task."""
        # Create model that returns valid responses for all components
        # User simulator, tool simulator, and evaluator all need JSON responses
        model = MACSModelAdapter(
            responses=[
                # User response
                '{"text": "Yes, please book that flight.", "details": {}}',
                # Evaluator responses (user GSR and system GSR)
                '[{"assertion": "user: Booking confirmed", "answer": "TRUE", "evidence": "User confirmed booking"}]',
                '[{"assertion": "agent: Database updated", "answer": "TRUE", "evidence": "Agent updated database"}]',
            ]
        )

        benchmark = ConcreteMACSBenchmark(sample_agent_data, model)
        reports = benchmark.run([sample_task])

        # Should have exactly one report
        assert len(reports) == 1

        report = reports[0]
        assert report["task_id"] == str(sample_task.id)
        assert report["repeat_idx"] == 0
        assert report["status"] == "success"
        assert "traces" in report
        assert "config" in report
        assert "eval" in report

    def test_run_multiple_tasks(self, sample_agent_data, macs_task_collection):
        """Run benchmark with multiple tasks via TaskCollection."""
        model = MACSModelAdapter(
            responses=[
                # Responses cycle for each task
                '{"text": "User response", "details": {}}',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
            ]
        )

        benchmark = ConcreteMACSBenchmark(sample_agent_data, model)
        reports = benchmark.run(macs_task_collection)

        # Should have one report per task
        assert len(reports) == len(macs_task_collection)

        # Each report should have correct structure
        for i, report in enumerate(reports):
            assert report["repeat_idx"] == 0
            assert report["status"] == "success"
            assert "eval" in report

    def test_run_with_task_repeats(self, sample_agent_data, sample_task):
        """Run benchmark with multiple task repetitions."""
        model = MACSModelAdapter(
            responses=[
                '{"text": "response", "details": {}}',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
                '[{"assertion": "test", "answer": "FALSE", "evidence": "failed"}]',
            ]
        )

        n_repeats = 3
        benchmark = ConcreteMACSBenchmark(sample_agent_data, model, n_task_repeats=n_repeats)
        reports = benchmark.run([sample_task])

        # Should have n_repeats reports for the single task
        assert len(reports) == n_repeats

        # Check repeat indices
        for i, report in enumerate(reports):
            assert report["repeat_idx"] == i
            assert report["task_id"] == str(sample_task.id)

    def test_run_returns_traces(self, sample_agent_data, sample_task):
        """Benchmark run collects and returns traces from all components."""
        model = MACSModelAdapter(
            responses=[
                '{"text": "User says yes", "details": {}}',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
            ]
        )

        benchmark = ConcreteMACSBenchmark(sample_agent_data, model)
        reports = benchmark.run([sample_task])

        report = reports[0]
        traces = report["traces"]

        # Should have traces from environment and user at minimum
        # (agents may or may not have traces depending on implementation)
        assert isinstance(traces, dict)

    def test_run_returns_evaluation_results(self, sample_agent_data, sample_task):
        """Benchmark run returns evaluation results with GSR scores."""
        # Configure model to return specific evaluation results
        model = MACSModelAdapter(
            responses=[
                '{"text": "User response", "details": {}}',
                # User GSR: 1/2 assertions TRUE
                '[{"assertion": "user: Booking confirmed", "answer": "TRUE", "evidence": "confirmed"}, '
                '{"assertion": "user: Something else", "answer": "FALSE", "evidence": "not found"}]',
                # System GSR: 1/1 assertions TRUE
                '[{"assertion": "agent: Database updated", "answer": "TRUE", "evidence": "updated"}]',
            ]
        )

        benchmark = ConcreteMACSBenchmark(sample_agent_data, model)
        reports = benchmark.run([sample_task])

        report = reports[0]
        eval_result = report["eval"]

        # Should have evaluation results (list of eval dicts)
        assert eval_result is not None
        assert isinstance(eval_result, list)

    def test_run_handles_evaluation_failure_gracefully(self, sample_agent_data, sample_task):
        """Benchmark continues even when evaluation fails."""
        # Model returns invalid JSON for evaluator
        model = MACSModelAdapter(
            responses=[
                '{"text": "User response", "details": {}}',
                "not valid json - evaluator will fail",
                "also not valid json",
            ]
        )

        benchmark = ConcreteMACSBenchmark(sample_agent_data, model)
        reports = benchmark.run([sample_task])

        # Should still get a report even though evaluation had issues
        assert len(reports) == 1
        report = reports[0]

        # Task should complete (status depends on error handling config)
        assert "status" in report

    def test_run_with_callbacks(self, sample_agent_data, sample_task):
        """Benchmark triggers callbacks during run."""
        from maseval import BenchmarkCallback

        class TrackingCallback(BenchmarkCallback):
            def __init__(self):
                self.events = []

            def on_run_start(self, benchmark):
                self.events.append("run_start")

            def on_task_start(self, benchmark, task):
                self.events.append("task_start")

            def on_task_repeat_start(self, benchmark, task, repeat_idx):
                self.events.append(f"repeat_start_{repeat_idx}")

            def on_task_repeat_end(self, benchmark, report):
                self.events.append(f"repeat_end_{report['repeat_idx']}")

            def on_task_end(self, benchmark, task, result):
                self.events.append("task_end")

            def on_run_end(self, benchmark, results):
                self.events.append("run_end")

        model = MACSModelAdapter(
            responses=[
                '{"text": "response", "details": {}}',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
            ]
        )

        callback = TrackingCallback()
        benchmark = ConcreteMACSBenchmark(sample_agent_data, model, callbacks=[callback])
        benchmark.run([sample_task])

        # Verify callback sequence
        assert "run_start" in callback.events
        assert "task_start" in callback.events
        assert "repeat_start_0" in callback.events
        assert "repeat_end_0" in callback.events
        assert "task_end" in callback.events
        assert "run_end" in callback.events

        # Verify order
        assert callback.events.index("run_start") < callback.events.index("task_start")
        assert callback.events.index("task_start") < callback.events.index("repeat_start_0")
        assert callback.events.index("repeat_end_0") < callback.events.index("task_end")
        assert callback.events.index("task_end") < callback.events.index("run_end")

    def test_run_computes_benchmark_metrics(self, sample_agent_data, sample_task):
        """Benchmark metrics can be computed from run results."""
        model = MACSModelAdapter(
            responses=[
                '{"text": "response", "details": {}}',
                '[{"assertion": "user: Booking confirmed", "answer": "TRUE", "evidence": "ok"}]',
                '[{"assertion": "agent: Database updated", "answer": "TRUE", "evidence": "ok"}]',
            ]
        )

        benchmark = ConcreteMACSBenchmark(sample_agent_data, model)
        reports = benchmark.run([sample_task])

        # compute_benchmark_metrics expects full reports (each with "eval" key)
        metrics = compute_benchmark_metrics(reports)
        assert isinstance(metrics, dict)
        assert "total_tasks" in metrics
        assert "success_rate" in metrics
        assert "mean_metrics" in metrics

    def test_full_pipeline_with_travel_task(self, sample_agent_data, travel_task):
        """Full end-to-end test with realistic travel task."""
        # Comprehensive responses for the full pipeline
        model = MACSModelAdapter(
            responses=[
                # User simulator response (acknowledging agent's response)
                '{"text": "Yes, that flight works for me. Please book it.", "details": {"satisfied": true}}',
                # User GSR evaluator response
                """[
                    {"assertion": "user: The user's flight booking request was acknowledged", "answer": "TRUE", "evidence": "Agent acknowledged the booking request"},
                    {"assertion": "user: The user received flight options or a confirmation", "answer": "TRUE", "evidence": "User confirmed the flight"}
                ]""",
                # System GSR evaluator response
                '[{"assertion": "agent: The search_flights tool was called with correct parameters", "answer": "TRUE", "evidence": "Tool was called"}]',
            ]
        )

        benchmark = ConcreteMACSBenchmark(sample_agent_data, model)
        reports = benchmark.run([travel_task])

        assert len(reports) == 1
        report = reports[0]

        # Verify successful execution
        assert report["status"] == "success"
        assert report["task_id"] == str(travel_task.id)

        # Verify traces were collected
        assert "traces" in report
        assert isinstance(report["traces"], dict)

        # Verify config was collected
        assert "config" in report
        assert isinstance(report["config"], dict)

        # Verify evaluation ran
        assert "eval" in report
