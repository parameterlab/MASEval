"""Test LLM Simulator functionality.

These tests verify that LLMSimulator retry logic and tracing work correctly.
"""

from typing import cast

import pytest
from maseval.core.simulator import (
    ToolLLMSimulator,
    SimulatorCallStatus,
    ToolSimulatorError,
)


@pytest.mark.core
class TestLLMSimulator:
    """Tests for LLMSimulator retry and tracing."""

    def test_llm_simulator_retry_logic(self, dummy_model):
        """Test that simulator retries on parsing errors."""
        # Model returns invalid JSON first, then valid
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(
            responses=[
                "invalid json",
                '{"text": "Tool executed successfully", "details": {"result": "success"}}',
            ]
        )

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        result = simulator(actual_inputs={"param": "test"})

        # Should eventually succeed - ToolLLMSimulator returns (text, details) tuple
        assert result is not None
        assert isinstance(result, tuple)
        text, details = result
        assert isinstance(details, dict)
        assert details.get("result") == "success"

        # Should have 2 attempts captured in logs (1 fail, 1 success)
        assert len(simulator.logs) == 2

    def test_llm_simulator_parsing_error_retry(self, dummy_model):
        """Test that parsing errors trigger retries and raise SimulatorError on exhaustion."""
        from conftest import DummyModelAdapter

        # All responses are invalid JSON
        model = DummyModelAdapter(responses=["bad", "bad", "bad"])

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        # Should raise ToolSimulatorError after max_try attempts
        with pytest.raises(ToolSimulatorError) as exc_info:
            simulator(actual_inputs={"param": "test"})

        # Verify exception details
        err = cast(ToolSimulatorError, exc_info.value)
        assert err.attempts == 3
        assert err.last_error is not None
        assert len(err.logs) == 3  # All 3 attempts in exception logs
        assert len(simulator.logs) == 3  # All 3 attempts logged in simulator

    def test_llm_simulator_max_attempts_respected(self, dummy_model):
        """Test that max_try limit is respected."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=["invalid"] * 10)

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=2,  # Only allow 2 attempts
        )

        # Should raise after 2 attempts
        with pytest.raises(ToolSimulatorError) as exc_info:
            simulator(actual_inputs={"param": "test"})

        # Should stop after 2 attempts, not continue to 10
        err = cast(ToolSimulatorError, exc_info.value)
        assert len(simulator.logs) == 2
        assert err.attempts == 2

    def test_llm_simulator_history_structure(self, dummy_model):
        """Test that history entries have correct structure."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"result": "success"}'])

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        _ = simulator(actual_inputs={"param": "test"})

        # Check log entry structure
        entry = simulator.logs[0]
        assert "id" in entry
        assert "timestamp" in entry
        assert "input" in entry
        assert "prompt" in entry
        assert "raw_output" in entry
        assert "parsed_output" in entry
        assert "status" in entry

    def test_llm_simulator_status_tracking(self, dummy_model):
        """Test that status is correctly tracked."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"result": "success"}'])

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        _ = simulator(actual_inputs={"param": "test"})

        entry = simulator.logs[0]
        assert entry["status"] == SimulatorCallStatus.Successful.value

    def test_llm_simulator_gather_traces(self, dummy_model):
        """Test that gather_traces includes complete history."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"result": "success"}'])

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        _ = simulator(actual_inputs={"param": "test"})

        traces = simulator.gather_traces()

        assert "simulator_type" in traces
        assert "total_calls" in traces
        assert "successful_calls" in traces
        assert "failed_calls" in traces
        assert "logs" in traces
        assert traces["successful_calls"] == 1


@pytest.mark.core
class TestUserLLMSimulatorValidation:
    """Tests for UserLLMSimulator early stopping validation."""

    def test_stop_token_without_condition_raises(self, dummy_model):
        """ValueError raised when stop_token set but early_stopping_condition is None."""
        from maseval.core.simulator import UserLLMSimulator

        with pytest.raises(ValueError, match="must both be set or both be None"):
            UserLLMSimulator(
                model=dummy_model,
                user_profile={"name": "test"},
                scenario="test scenario",
                stop_token="</stop>",
            )

    def test_condition_without_stop_token_raises(self, dummy_model):
        """ValueError raised when early_stopping_condition set but stop_token is None."""
        from maseval.core.simulator import UserLLMSimulator

        with pytest.raises(ValueError, match="must both be set or both be None"):
            UserLLMSimulator(
                model=dummy_model,
                user_profile={"name": "test"},
                scenario="test scenario",
                early_stopping_condition="goals are met",
            )

    def test_both_none_is_valid(self, dummy_model):
        """No error when both stop_token and early_stopping_condition are None."""
        from maseval.core.simulator import UserLLMSimulator

        simulator = UserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "test"},
            scenario="test scenario",
        )
        assert simulator.stop_token is None
        assert simulator.early_stopping_condition is None

    def test_both_set_is_valid(self, dummy_model):
        """No error when both stop_token and early_stopping_condition are set."""
        from maseval.core.simulator import UserLLMSimulator

        simulator = UserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "test"},
            scenario="test scenario",
            stop_token="</stop>",
            early_stopping_condition="all goals accomplished",
        )
        assert simulator.stop_token == "</stop>"
        assert simulator.early_stopping_condition == "all goals accomplished"
