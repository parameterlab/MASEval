"""Integration tests for smolagents.

These tests require smolagents to be installed.
Run with: pytest -m smolagents
"""

import pytest

# Skip entire module if smolagents not installed
pytest.importorskip("smolagents")

# Mark all tests in this file as requiring smolagents
pytestmark = [pytest.mark.interface, pytest.mark.smolagents]


def test_smolagents_wrapper_import():
    """Test that SmolAgentAdapter can be imported when smolagents is installed."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter, SmolAgentUser

    assert SmolAgentAdapter is not None
    assert SmolAgentUser is not None


def test_smolagents_in_agents_all():
    """Test that smolagents appears in interface.agents.__all__ when installed."""
    import maseval.interface.agents

    assert "SmolAgentAdapter" in maseval.interface.agents.__all__
    assert "SmolAgentUser" in maseval.interface.agents.__all__


def test_check_smolagents_installed_function():
    """Test that _check_smolagents_installed doesn't raise when smolagents is installed."""
    from maseval.interface.agents.smolagents import _check_smolagents_installed

    # Should not raise
    _check_smolagents_installed()


def test_smolagents_wrapper_creation():
    """Test that SmolAgentAdapter can be created."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter

    # Create wrapper with mock agent
    wrapper = SmolAgentAdapter(agent_instance=object(), name="test_agent")

    assert wrapper.name == "test_agent"
    assert wrapper.agent is not None


def test_smolagents_user_creation():
    """Test that SmolAgentUser can be created."""
    from maseval.interface.agents.smolagents import SmolAgentUser
    from unittest.mock import Mock

    # Create user with required parameters
    mock_model = Mock()
    user = SmolAgentUser(
        name="test_user",
        model=mock_model,
        user_profile={"role": "tester"},
        scenario="test scenario",
        initial_prompt="test prompt",
    )

    assert user is not None
    assert user.name == "test_user"


def test_smolagents_wrapper_gather_traces_with_monitoring():
    """Test that SmolAgentAdapter.gather_traces() captures token and timing data."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import ActionStep, AgentMemory
    from smolagents.monitoring import TokenUsage, Timing
    from unittest.mock import Mock
    import time

    # Create a mock agent with memory
    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test system prompt")

    # Add some ActionSteps with monitoring data
    start_time = time.time()

    # Step 1: ActionStep with token usage and timing
    step1 = ActionStep(
        step_number=1,
        timing=Timing(start_time=start_time, end_time=start_time + 0.5),
        observations_images=[],
    )
    step1.token_usage = TokenUsage(input_tokens=100, output_tokens=50)
    step1.observations = "Observation from step 1"
    step1.action_output = "Output from step 1"
    mock_agent.memory.steps.append(step1)

    # Step 2: Another ActionStep
    step2 = ActionStep(
        step_number=2,
        timing=Timing(start_time=start_time + 0.5, end_time=start_time + 1.2),
        observations_images=[],
    )
    step2.token_usage = TokenUsage(input_tokens=200, output_tokens=100)
    step2.observations = "Observation from step 2"
    step2.action_output = "Output from step 2"
    mock_agent.memory.steps.append(step2)

    # Mock write_memory_to_messages to return empty list (we're testing gather_traces, not get_messages)
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    # Create wrapper
    wrapper = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Call gather_traces
    traces = wrapper.gather_traces()

    # Verify aggregated statistics
    assert "total_steps" in traces
    assert traces["total_steps"] == 2

    assert "total_input_tokens" in traces
    assert traces["total_input_tokens"] == 300  # 100 + 200

    assert "total_output_tokens" in traces
    assert traces["total_output_tokens"] == 150  # 50 + 100

    assert "total_tokens" in traces
    assert traces["total_tokens"] == 450  # 300 + 150

    assert "total_duration_seconds" in traces
    assert traces["total_duration_seconds"] == pytest.approx(1.2, abs=0.01)  # 0.5 + 0.7

    # Verify step details
    assert "steps_detail" in traces
    assert len(traces["steps_detail"]) == 2

    # Check step 1 details
    step1_detail = traces["steps_detail"][0]
    assert step1_detail["step_number"] == 1
    assert step1_detail["input_tokens"] == 100
    assert step1_detail["output_tokens"] == 50
    assert step1_detail["total_tokens"] == 150
    assert step1_detail["duration_seconds"] == pytest.approx(0.5, abs=0.01)
    assert step1_detail["observations"] == "Observation from step 1"
    assert step1_detail["action_output"] == "Output from step 1"

    # Check step 2 details
    step2_detail = traces["steps_detail"][1]
    assert step2_detail["step_number"] == 2
    assert step2_detail["input_tokens"] == 200
    assert step2_detail["output_tokens"] == 100
    assert step2_detail["total_tokens"] == 300
    assert step2_detail["duration_seconds"] == pytest.approx(0.7, abs=0.01)
    assert step2_detail["observations"] == "Observation from step 2"
    assert step2_detail["action_output"] == "Output from step 2"


def test_smolagents_wrapper_gather_traces_without_monitoring():
    """Test that gather_traces works when agent has no monitoring data."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import AgentMemory
    from unittest.mock import Mock

    # Create a mock agent with empty memory
    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test system prompt")
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    # Create wrapper
    wrapper = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Call gather_traces
    traces = wrapper.gather_traces()

    # Verify aggregated statistics show zero usage
    assert "total_steps" in traces
    assert traces["total_steps"] == 0

    assert "total_input_tokens" in traces
    assert traces["total_input_tokens"] == 0

    assert "total_output_tokens" in traces
    assert traces["total_output_tokens"] == 0

    assert "total_tokens" in traces
    assert traces["total_tokens"] == 0

    assert "total_duration_seconds" in traces
    assert traces["total_duration_seconds"] == 0.0

    assert "steps_detail" in traces
    assert len(traces["steps_detail"]) == 0


def test_smolagents_wrapper_gather_traces_with_planning_step():
    """Test that gather_traces captures PlanningStep data correctly."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import PlanningStep, AgentMemory
    from smolagents.monitoring import TokenUsage, Timing
    from smolagents.models import ChatMessage, MessageRole
    from unittest.mock import Mock
    import time

    # Create a mock agent with memory
    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test system prompt")

    # Add a PlanningStep
    start_time = time.time()
    planning_step = PlanningStep(
        timing=Timing(start_time=start_time, end_time=start_time + 1.0),
        model_input_messages=[],
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Planning output"),
        plan="This is my plan",
    )
    planning_step.token_usage = TokenUsage(input_tokens=500, output_tokens=200)
    mock_agent.memory.steps.append(planning_step)

    # Mock write_memory_to_messages
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    # Create wrapper
    wrapper = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Call gather_traces
    traces = wrapper.gather_traces()

    # Verify aggregated statistics
    assert traces["total_steps"] == 1
    assert traces["total_input_tokens"] == 500
    assert traces["total_output_tokens"] == 200
    assert traces["total_tokens"] == 700
    assert traces["total_duration_seconds"] == pytest.approx(1.0, abs=0.01)

    # Verify step details
    assert len(traces["steps_detail"]) == 1
    step_detail = traces["steps_detail"][0]
    # PlanningStep may not have step_number, so it could be None
    assert step_detail["input_tokens"] == 500
    assert step_detail["output_tokens"] == 200
    assert step_detail["total_tokens"] == 700
    assert step_detail["duration_seconds"] == pytest.approx(1.0, abs=0.01)
    assert step_detail["plan"] == "This is my plan"
    # PlanningStep should not have action_output or observations
    assert "action_output" not in step_detail
    assert "observations" not in step_detail


def test_smolagents_wrapper_message_manipulation_not_supported():
    """Test that smolagents explicitly raises NotImplementedError for message manipulation.

    smolagents builds its AgentMemory from execution steps and does not support
    arbitrary message injection. The wrapper should raise clear NotImplementedError
    for set_message_history and append_to_message_history operations.

    Only clear_message_history is supported (resets memory with system prompt).
    """
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from maseval import MessageHistory
    from smolagents import CodeAgent
    from conftest import FakeSmolagentsModel

    # Create a smolagents agent
    mock_model = FakeSmolagentsModel(["Test response"])
    agent = CodeAgent(tools=[], model=mock_model, max_steps=1)
    wrapper = SmolAgentAdapter(agent_instance=agent, name="test_agent")

    # Test that append_to_message_history raises NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        wrapper.append_to_message_history("user", "Manual message")

    assert "doesn't support appending" in str(exc_info.value)
    assert "memory is built from execution steps" in str(exc_info.value)

    # Test that set_message_history raises NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        new_history = MessageHistory()
        new_history.add_message("user", "Test message")
        wrapper.set_message_history(new_history)

    assert "doesn't support setting" in str(exc_info.value)
    assert "memory is built from execution steps" in str(exc_info.value)


def test_smolagents_wrapper_clear_message_history_supported():
    """Test that smolagents supports clear_message_history.

    clear_message_history is the only history manipulation operation
    supported by smolagents. It resets the AgentMemory while preserving
    the system prompt.
    """
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents import CodeAgent
    from conftest import FakeSmolagentsModel

    # Create a smolagents agent
    mock_model = FakeSmolagentsModel(["Test response"])
    agent = CodeAgent(tools=[], model=mock_model, max_steps=1)
    wrapper = SmolAgentAdapter(agent_instance=agent, name="test_agent")

    # Run the agent to populate memory
    wrapper.run("Test query")

    # Verify memory has content (should have multiple messages after run)
    messages_before = wrapper.get_messages()
    assert len(messages_before) > 1  # At least system + user messages

    # Clear the memory
    wrapper.clear_message_history()

    # Verify memory is reset (only system message remains)
    messages_after = wrapper.get_messages()
    assert len(messages_after) == 1
    assert messages_after[0]["role"] == "system"
    # System prompt content is framework-specific, just verify it exists and has content
    assert len(messages_after[0]["content"]) > 0
