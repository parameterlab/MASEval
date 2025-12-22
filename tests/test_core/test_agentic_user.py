"""Tests for AgenticUser and AgenticUserLLMSimulator."""

import pytest
import json
from unittest.mock import MagicMock
from maseval.core.agentic_user import AgenticUser
from maseval.core.model import ModelAdapter


class MockModelAdapter(ModelAdapter):
    def __init__(self, responses):
        super().__init__()
        self.responses = responses
        self.call_count = 0

    @property
    def model_id(self):
        return "mock-model"

    def _generate_impl(self, prompt: str, generation_params=None, **kwargs):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class DummyAgenticUser(AgenticUser):
    def get_tool(self):
        return None


@pytest.fixture
def mock_tool():
    tool = MagicMock(return_value="Tool executed successfully")
    tool.__doc__ = "A mock tool."
    return tool


@pytest.mark.core
def test_agentic_user_tool_usage(mock_tool):
    """Test that AgenticUser can execute a tool loop."""

    # Mock model responses:
    # 1. Thought + Tool Call
    # 2. Final response (after tool execution)

    response_1 = json.dumps({"text": "I need to check something.", "tool_calls": [{"name": "mock_tool", "arguments": {"arg": "val"}}]})

    response_2 = json.dumps({"text": "The tool worked.", "tool_calls": []})

    model = MockModelAdapter([response_1, response_2])

    user = DummyAgenticUser(name="AgenticTester", model=model, user_profile={}, scenario="Testing tools", tools={"mock_tool": mock_tool})

    # Simulate response
    final_response = user.simulate_response("Please check.")

    # Verification
    assert final_response == "The tool worked."
    assert mock_tool.called
    assert mock_tool.call_args[1] == {"arg": "val"}

    # Check history
    # The shared history should only contain the final response
    assert len(user.messages) == 2
    assert user.messages[-1]["content"] == "The tool worked."


@pytest.mark.core
def test_agentic_user_max_steps(mock_tool):
    """Test that AgenticUser respects max_internal_steps."""

    # Always return tool call
    response = json.dumps({"text": "Looping...", "tool_calls": [{"name": "mock_tool", "arguments": {}}]})

    model = MockModelAdapter([response])

    user = DummyAgenticUser(
        name="Looper", model=model, user_profile={}, scenario="Infinite loop", tools={"mock_tool": mock_tool}, max_internal_steps=3
    )

    final_response = user.simulate_response("Go.")

    assert "I need to stop" in final_response
    assert mock_tool.call_count == 3
