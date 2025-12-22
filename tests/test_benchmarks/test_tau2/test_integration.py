"""Integration tests for Tau2 Benchmark."""

import pytest
from unittest.mock import MagicMock

from maseval import AgentAdapter, Task, ModelAdapter
from maseval.benchmark.tau2 import Tau2Benchmark, Tau2Evaluator


class MockAgent(AgentAdapter):
    def get_messages(self):
        return []

    def _run_agent(self, query):
        return "Processed"


class MockModel(ModelAdapter):
    @property
    def model_id(self):
        return "mock"

    def _generate_impl(self, prompt, generation_params=None, **kwargs):
        # Return a valid JSON for agentic user
        return '{"text": "I am satisfied.", "tool_calls": []}'


class IntegrationTau2Benchmark(Tau2Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        agent = MockAgent(agent_instance=None, name="TestAgent")
        return [agent], {"TestAgent": agent}

    def get_model_adapter(self, model_id, **kwargs):
        return MockModel()


@pytest.mark.benchmark
def test_tau2_dry_run():
    """Smoke test for a full benchmark run with mocks."""

    # Setup task
    task = MagicMock(spec=Task)
    task.id = "test_1"
    task.environment_data = {"domain": "retail"}
    task.user_data = {"model_id": "mock-user", "instructions": "Test scenario"}
    task.evaluation_data = {"reward_basis": ["DB"], "actions": []}
    task.query = "Help me."

    # Setup benchmark
    benchmark = IntegrationTau2Benchmark(agent_data={}, n_task_repeats=1)

    # Mock Environment
    env_mock = MagicMock()
    env_mock.domain = "retail"
    env_mock.get_db_hash.return_value = "hash1"
    env_mock.create_user_tools.return_value = {}
    env_mock.gather_traces.return_value = {"final_db_hash": "hash1"}

    # Mock Evaluator to avoid real DB/Logic
    mock_evaluator = MagicMock(spec=Tau2Evaluator)
    mock_evaluator.return_value = {
        "reward": 1.0,
        "passed": True,
        "reward_breakdown": {"db": 1.0},
        "env_check": {},
        "action_check": {},
        "communicate_check": {},
    }
    # Mock filter_traces to return something valid
    mock_evaluator.filter_traces.return_value = {"termination_reason": "agent_stop"}

    # Patch setup_environment
    benchmark.setup_environment = MagicMock(return_value=env_mock)  # type: ignore[assignment]

    # Patch setup_evaluators to return our mock
    benchmark.setup_evaluators = MagicMock(return_value=[mock_evaluator])  # type: ignore[assignment]
    results = benchmark.run([task])

    # Debug info if failed
    if results[0]["status"] != "success":
        print(f"FAILED RESULT: {results[0]}")

    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert "eval" in results[0]