"""Integration tests for CAMEL-AI.

These tests require camel-ai to be installed.
Run with: pytest -m camel
"""

import pytest

# Skip entire module if camel-ai not installed
pytest.importorskip("camel")

# Mark all tests in this file as requiring camel
pytestmark = [pytest.mark.interface, pytest.mark.camel]


def test_camel_adapter_import():
    """Test that CamelAgentAdapter can be imported when camel-ai is installed."""
    from maseval.interface.agents.camel import CamelAgentAdapter, CamelUser

    assert CamelAgentAdapter is not None
    assert CamelUser is not None


def test_camel_in_agents_all():
    """Test that camel appears in interface.agents.__all__ when installed."""
    import maseval.interface.agents

    assert "CamelAgentAdapter" in maseval.interface.agents.__all__
    assert "CamelUser" in maseval.interface.agents.__all__


def test_check_camel_installed_function():
    """Test that _check_camel_installed doesn't raise when camel-ai is installed."""
    from maseval.interface.agents.camel import _check_camel_installed

    # Should not raise
    _check_camel_installed()


def test_camel_adapter_creation():
    """Test that CamelAgentAdapter can be created with a mock agent."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    # Create adapter with mock agent
    agent_adapter = CamelAgentAdapter(agent_instance=object(), name="test_agent")

    assert agent_adapter.name == "test_agent"
    assert agent_adapter.agent is not None


def test_camel_user_creation():
    """Test that CamelUser can be created."""
    from maseval.interface.agents.camel import CamelUser
    from unittest.mock import Mock

    # Create user with required parameters
    mock_model = Mock()
    user = CamelUser(
        name="test_user",
        model=mock_model,
        user_profile={"role": "tester"},
        scenario="test scenario",
        initial_query="test prompt",
    )

    assert user is not None
    assert user.name == "test_user"


def test_camel_adapter_gather_traces_basic():
    """Test that CamelAgentAdapter.gather_traces() returns expected base fields."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from unittest.mock import Mock

    # Create a mock agent with memory
    mock_agent = Mock()
    mock_agent.memory = None  # No memory configured

    # Create adapter
    agent_adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Call gather_traces
    traces = agent_adapter.gather_traces()

    # Verify base fields are present
    assert "type" in traces
    assert "gathered_at" in traces
    assert "name" in traces
    assert traces["name"] == "test_agent"
    assert "agent_type" in traces
    assert "messages" in traces
    assert "logs" in traces


def test_camel_adapter_gather_traces_with_response():
    """Test that gather_traces captures response metadata when available."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from unittest.mock import Mock

    # Create a mock agent
    mock_agent = Mock()
    mock_agent.memory = None

    # Create adapter
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Simulate a response being cached
    mock_response = Mock()
    mock_response.terminated = True
    mock_response.info = {"key": "value"}
    adapter._last_response = mock_response

    # Call gather_traces
    traces = adapter.gather_traces()

    # Verify response metadata is captured
    assert "last_response_terminated" in traces
    assert traces["last_response_terminated"] is True
    assert "last_response_info" in traces


def test_camel_adapter_gather_config_basic():
    """Test that CamelAgentAdapter.gather_config() returns expected fields."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from unittest.mock import Mock

    # Create a mock agent
    mock_agent = Mock()
    mock_agent.memory = None
    mock_agent.system_message = None
    mock_agent.model = None
    mock_agent.tools = None

    # Create adapter
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Call gather_config
    config = adapter.gather_config()

    # Verify base fields
    assert "type" in config
    assert "gathered_at" in config
    assert "name" in config
    assert config["name"] == "test_agent"
    assert "agent_type" in config
    assert "adapter_type" in config
    assert config["adapter_type"] == "CamelAgentAdapter"


def test_camel_adapter_gather_config_with_system_message():
    """Test that gather_config captures system message."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from unittest.mock import Mock

    # Create a mock agent with system message
    mock_agent = Mock()
    mock_agent.memory = None
    mock_agent.model = None
    mock_agent.tools = None

    # Mock system message as object with content
    mock_sys_msg = Mock()
    mock_sys_msg.content = "You are a helpful assistant."
    mock_agent.system_message = mock_sys_msg

    # Create adapter
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Call gather_config
    config = adapter.gather_config()

    # Verify camel_config includes system message
    assert "camel_config" in config
    assert "system_message" in config["camel_config"]
    assert config["camel_config"]["system_message"] == "You are a helpful assistant."


def test_camel_adapter_gather_config_with_tools():
    """Test that gather_config captures tool information."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from unittest.mock import Mock

    # Create mock tools
    mock_tool1 = Mock()
    mock_tool1.name = "search"
    mock_tool1.description = "Search the web"

    mock_tool2 = Mock()
    mock_tool2.name = "calculator"
    mock_tool2.description = "Perform calculations"

    # Create a mock agent with tools
    mock_agent = Mock()
    mock_agent.memory = None
    mock_agent.system_message = None
    mock_agent.model = None
    mock_agent.tools = [mock_tool1, mock_tool2]

    # Create adapter
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Call gather_config
    config = adapter.gather_config()

    # Verify tools are captured
    assert "camel_config" in config
    assert "tools" in config["camel_config"]
    assert len(config["camel_config"]["tools"]) == 2
    assert config["camel_config"]["tools"][0]["name"] == "search"
    assert config["camel_config"]["tools"][1]["name"] == "calculator"


def test_camel_adapter_get_messages_empty():
    """Test that get_messages returns empty MessageHistory when no messages."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from maseval.core.history import MessageHistory
    from unittest.mock import Mock

    # Create a mock agent with no memory
    mock_agent = Mock()
    mock_agent.memory = None

    # Create adapter
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Get messages
    messages = adapter.get_messages()

    # Verify it's a MessageHistory (possibly empty)
    assert isinstance(messages, MessageHistory)
    assert len(messages) == 0


def test_camel_adapter_get_messages_with_memory():
    """Test that get_messages correctly converts memory messages."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from maseval.core.history import MessageHistory
    from unittest.mock import Mock

    # Create mock memory with messages
    mock_memory = Mock()
    mock_memory.get_context.return_value = (
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        50,  # token count
    )

    # Create a mock agent with memory
    mock_agent = Mock()
    mock_agent.memory = mock_memory

    # Create adapter
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Get messages
    messages = adapter.get_messages()

    # Verify messages are converted
    assert isinstance(messages, MessageHistory)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi there!"


def test_camel_adapter_logs_initially_empty():
    """Test that adapter logs are initially empty."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.memory = None

    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    assert isinstance(adapter.logs, list)
    assert len(adapter.logs) == 0


def test_camel_user_get_tool():
    """Test that CamelUser.get_tool() returns a CAMEL FunctionTool."""
    from maseval.interface.agents.camel import CamelUser
    from camel.toolkits import FunctionTool
    from unittest.mock import Mock

    # Create mock model
    mock_model = Mock()

    # Create user
    user = CamelUser(
        name="test_user",
        model=mock_model,
        user_profile={"role": "customer"},
        scenario="Testing scenario",
        initial_query="Hello",
    )

    # Get the tool
    tool = user.get_tool()

    # Verify it's a FunctionTool
    assert isinstance(tool, FunctionTool)


def test_camel_adapter_extract_final_answer():
    """Test that _extract_final_answer correctly extracts content from response."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.memory = None

    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Create mock response with msgs
    mock_msg = Mock()
    mock_msg.content = "This is the answer"

    mock_response = Mock()
    mock_response.msgs = [mock_msg]

    # Test extraction
    answer = adapter._extract_final_answer(mock_response)
    assert answer == "This is the answer"


def test_camel_adapter_extract_final_answer_from_msg():
    """Test extraction when response has msg instead of msgs."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.memory = None

    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Create mock response with singular msg
    mock_msg = Mock()
    mock_msg.content = "Single message answer"

    mock_response = Mock()
    mock_response.msgs = []  # Empty msgs
    mock_response.msg = mock_msg

    # Test extraction
    answer = adapter._extract_final_answer(mock_response)
    assert answer == "Single message answer"


def test_camel_adapter_convert_memory_messages_with_tool_calls():
    """Test that tool-related fields are preserved during conversion."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.memory = None

    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Messages with tool calls
    memory_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "Search results", "tool_call_id": "call_1", "name": "search"},
    ]

    # Convert
    converted = adapter._convert_memory_messages(memory_messages)

    # Verify tool-related fields are preserved
    assert len(converted) == 2
    assert "tool_calls" in converted[0]
    assert converted[1]["role"] == "tool"
    assert converted[1]["tool_call_id"] == "call_1"
    assert converted[1]["name"] == "search"


# =============================================================================
# Phase 2 Tests: CamelAgentUser, Execution Loop, Tracers
# =============================================================================


def test_camel_agent_user_import():
    """Test that CamelAgentUser can be imported."""
    from maseval.interface.agents.camel import CamelAgentUser

    assert CamelAgentUser is not None


def test_camel_agent_user_in_all():
    """Test that CamelAgentUser is in __all__."""
    import maseval.interface.agents

    assert "CamelAgentUser" in maseval.interface.agents.__all__


def test_camel_agent_user_creation():
    """Test CamelAgentUser creation with a mock agent."""
    from maseval.interface.agents.camel import CamelAgentUser
    from maseval.core.user import User
    from unittest.mock import Mock

    mock_agent = Mock()

    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello, I need help",
        name="test_user",
        max_turns=5,
    )

    assert user.name == "test_user"
    assert user._max_turns == 5
    assert user._turn_count == 0
    assert isinstance(user, User)


def test_camel_agent_user_get_initial_query():
    """Test CamelAgentUser.get_initial_query() returns the initial query."""
    from maseval.interface.agents.camel import CamelAgentUser
    from unittest.mock import Mock

    mock_agent = Mock()
    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Test initial query",
    )

    assert user.get_initial_query() == "Test initial query"


def test_camel_agent_user_is_done():
    """Test CamelAgentUser.is_done() respects max_turns."""
    from maseval.interface.agents.camel import CamelAgentUser
    from unittest.mock import Mock

    mock_agent = Mock()
    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
        max_turns=2,
    )

    assert user.is_done() is False

    user._turn_count = 1
    assert user.is_done() is False

    user._turn_count = 2
    assert user.is_done() is True


def test_camel_agent_user_respond():
    """Test CamelAgentUser.respond() delegates to the CAMEL agent."""
    from maseval.interface.agents.camel import CamelAgentUser
    from unittest.mock import Mock

    # Create mock agent with response
    mock_msg = Mock()
    mock_msg.content = "Agent response"

    mock_response = Mock()
    mock_response.msgs = [mock_msg]

    mock_agent = Mock()
    mock_agent.step.return_value = mock_response

    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
        max_turns=5,
    )

    response = user.respond("What is your question?")

    assert response == "Agent response"
    assert user._turn_count == 1
    assert mock_agent.step.called


def test_camel_agent_user_respond_when_done():
    """Test CamelAgentUser.respond() returns empty string when done."""
    from maseval.interface.agents.camel import CamelAgentUser
    from unittest.mock import Mock

    mock_agent = Mock()
    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
        max_turns=1,
    )

    # Set as done
    user._turn_count = 1

    response = user.respond("Question?")

    assert response == ""
    assert not mock_agent.step.called


def test_camel_agent_user_get_tool():
    """Test CamelAgentUser.get_tool() returns a FunctionTool."""
    from maseval.interface.agents.camel import CamelAgentUser
    from camel.toolkits import FunctionTool
    from unittest.mock import Mock

    mock_agent = Mock()
    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
    )

    tool = user.get_tool()

    assert isinstance(tool, FunctionTool)


def test_camel_agent_user_gather_traces():
    """Test CamelAgentUser.gather_traces() returns expected fields."""
    from maseval.interface.agents.camel import CamelAgentUser
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.__class__.__name__ = "MockChatAgent"

    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
        name="test_user",
        max_turns=5,
    )

    traces = user.gather_traces()

    assert traces["type"] == "CamelAgentUser"
    assert traces["name"] == "test_user"
    assert traces["max_turns"] == 5
    assert traces["turns_used"] == 0
    assert "gathered_at" in traces
    assert "logs" in traces


def test_camel_agent_user_gather_config():
    """Test CamelAgentUser.gather_config() returns expected fields."""
    from maseval.interface.agents.camel import CamelAgentUser
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.__class__.__name__ = "MockChatAgent"

    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Test query",
        name="test_user",
        max_turns=10,
    )

    config = user.gather_config()

    assert config["type"] == "CamelAgentUser"
    assert config["name"] == "test_user"
    assert config["max_turns"] == 10
    assert "initial_query" in config


# =============================================================================
# Execution Loop Tests
# =============================================================================


def test_execution_loop_import():
    """Test that camel_role_playing_execution_loop can be imported."""
    from maseval.interface.agents.camel import camel_role_playing_execution_loop

    assert camel_role_playing_execution_loop is not None


def test_execution_loop_in_all():
    """Test that camel_role_playing_execution_loop is in __all__."""
    import maseval.interface.agents

    assert "camel_role_playing_execution_loop" in maseval.interface.agents.__all__


def test_execution_loop_basic():
    """Test camel_role_playing_execution_loop with mock RolePlaying."""
    from maseval.interface.agents.camel import camel_role_playing_execution_loop
    from unittest.mock import Mock

    # Create mock responses
    mock_msg = Mock()
    mock_msg.content = "Final answer"

    mock_assistant_response = Mock()
    mock_assistant_response.msgs = [mock_msg]
    mock_assistant_response.terminated = True

    mock_user_response = Mock()
    mock_user_response.terminated = False

    # Create mock RolePlaying
    mock_role_playing = Mock()
    mock_role_playing.step.return_value = (mock_assistant_response, mock_user_response)

    # Create mock task
    mock_task = Mock()

    # Execute
    result = camel_role_playing_execution_loop(
        mock_role_playing,
        mock_task,
        max_steps=5,
    )

    assert result == "Final answer"
    assert mock_role_playing.step.call_count == 1  # Should stop after first step (terminated)


def test_execution_loop_with_tracer():
    """Test camel_role_playing_execution_loop records steps in tracer."""
    from maseval.interface.agents.camel import camel_role_playing_execution_loop, CamelRolePlayingTracer
    from unittest.mock import Mock

    # Create mock responses
    mock_msg = Mock()
    mock_msg.content = "Answer"

    mock_assistant_response = Mock()
    mock_assistant_response.msgs = [mock_msg]
    mock_assistant_response.terminated = True

    mock_user_response = Mock()
    mock_user_response.terminated = False

    # Create mock RolePlaying
    mock_role_playing = Mock()
    mock_role_playing.step.return_value = (mock_assistant_response, mock_user_response)

    # Create tracer
    tracer = CamelRolePlayingTracer(mock_role_playing, name="test_tracer")

    # Execute with tracer
    camel_role_playing_execution_loop(
        mock_role_playing,
        Mock(),
        max_steps=5,
        tracer=tracer,
    )

    # Verify tracer recorded the step
    assert tracer._step_count == 1
    assert tracer._termination_reason == "assistant_terminated"


def test_execution_loop_max_steps():
    """Test camel_role_playing_execution_loop respects max_steps."""
    from maseval.interface.agents.camel import camel_role_playing_execution_loop
    from unittest.mock import Mock

    # Create mock responses that never terminate
    mock_msg = Mock()
    mock_msg.content = "Non-terminal answer"

    mock_assistant_response = Mock()
    mock_assistant_response.msgs = [mock_msg]
    mock_assistant_response.terminated = False

    mock_user_response = Mock()
    mock_user_response.terminated = False

    mock_role_playing = Mock()
    mock_role_playing.step.return_value = (mock_assistant_response, mock_user_response)

    # Execute with max_steps=3
    camel_role_playing_execution_loop(
        mock_role_playing,
        Mock(),
        max_steps=3,
    )

    # Should have executed exactly 3 steps
    assert mock_role_playing.step.call_count == 3


# =============================================================================
# RolePlaying Tracer Tests
# =============================================================================


def test_role_playing_tracer_import():
    """Test that CamelRolePlayingTracer can be imported."""
    from maseval.interface.agents.camel import CamelRolePlayingTracer

    assert CamelRolePlayingTracer is not None


def test_role_playing_tracer_in_all():
    """Test that CamelRolePlayingTracer is in __all__."""
    import maseval.interface.agents

    assert "CamelRolePlayingTracer" in maseval.interface.agents.__all__


def test_role_playing_tracer_creation():
    """Test CamelRolePlayingTracer creation."""
    from maseval.interface.agents.camel import CamelRolePlayingTracer
    from unittest.mock import Mock

    mock_role_playing = Mock()
    tracer = CamelRolePlayingTracer(mock_role_playing, name="test_tracer")

    assert tracer.name == "test_tracer"
    assert tracer._step_count == 0
    assert tracer._termination_reason is None


def test_role_playing_tracer_record_step():
    """Test CamelRolePlayingTracer.record_step() tracks progress."""
    from maseval.interface.agents.camel import CamelRolePlayingTracer
    from unittest.mock import Mock

    mock_role_playing = Mock()
    tracer = CamelRolePlayingTracer(mock_role_playing)

    # Create mock responses
    mock_assistant = Mock()
    mock_assistant.terminated = False

    mock_user = Mock()
    mock_user.terminated = False

    # Record a step
    tracer.record_step(mock_assistant, mock_user)

    assert tracer._step_count == 1
    assert tracer._termination_reason is None

    # Record another step with termination
    mock_assistant.terminated = True
    tracer.record_step(mock_assistant, mock_user)

    assert tracer._step_count == 2
    assert tracer._termination_reason == "assistant_terminated"


def test_role_playing_tracer_gather_traces():
    """Test CamelRolePlayingTracer.gather_traces() returns expected fields."""
    from maseval.interface.agents.camel import CamelRolePlayingTracer
    from unittest.mock import Mock

    mock_role_playing = Mock()
    tracer = CamelRolePlayingTracer(mock_role_playing, name="test_tracer")

    # Record some steps
    mock_response = Mock()
    mock_response.terminated = False
    tracer.record_step(mock_response, mock_response)
    tracer.record_step(mock_response, mock_response)

    traces = tracer.gather_traces()

    assert traces["name"] == "test_tracer"
    assert traces["trace_type"] == "role_playing_orchestration"
    assert traces["step_count"] == 2
    assert "step_logs" in traces
    assert len(traces["step_logs"]) == 2


def test_role_playing_tracer_gather_config():
    """Test CamelRolePlayingTracer.gather_config() returns expected fields."""
    from maseval.interface.agents.camel import CamelRolePlayingTracer
    from unittest.mock import Mock

    mock_role_playing = Mock()
    mock_role_playing.task_prompt = "Test task prompt"
    mock_role_playing.assistant_role_name = "Assistant"
    mock_role_playing.user_role_name = "User"

    tracer = CamelRolePlayingTracer(mock_role_playing, name="test_tracer")

    config = tracer.gather_config()

    assert config["name"] == "test_tracer"
    assert config["task_prompt"] == "Test task prompt"
    assert config["assistant_role"] == "Assistant"
    assert config["user_role"] == "User"


# =============================================================================
# Workforce Tracer Tests
# =============================================================================


def test_workforce_tracer_import():
    """Test that CamelWorkforceTracer can be imported."""
    from maseval.interface.agents.camel import CamelWorkforceTracer

    assert CamelWorkforceTracer is not None


def test_workforce_tracer_in_all():
    """Test that CamelWorkforceTracer is in __all__."""
    import maseval.interface.agents

    assert "CamelWorkforceTracer" in maseval.interface.agents.__all__


def test_workforce_tracer_creation():
    """Test CamelWorkforceTracer creation."""
    from maseval.interface.agents.camel import CamelWorkforceTracer
    from unittest.mock import Mock

    mock_workforce = Mock()
    tracer = CamelWorkforceTracer(mock_workforce, name="test_workforce")

    assert tracer.name == "test_workforce"


def test_workforce_tracer_gather_traces():
    """Test CamelWorkforceTracer.gather_traces() returns expected fields."""
    from maseval.interface.agents.camel import CamelWorkforceTracer
    from unittest.mock import Mock

    # Create mock workforce with internal state
    mock_workforce = Mock()
    mock_workforce._assignees = {"task_1": "worker_a", "task_2": "worker_b"}
    mock_workforce._pending_tasks = []
    mock_workforce._completed_tasks = []
    mock_workforce._task_dependencies = {"task_1": [], "task_2": ["task_1"]}

    tracer = CamelWorkforceTracer(mock_workforce, name="test_workforce")

    traces = tracer.gather_traces()

    assert traces["name"] == "test_workforce"
    assert traces["trace_type"] == "workforce_orchestration"
    assert "task_decomposition" in traces
    assert "worker_assignments" in traces
    assert "completed_tasks" in traces
    assert traces["pending_tasks_count"] == 0


def test_workforce_tracer_gather_config():
    """Test CamelWorkforceTracer.gather_config() returns expected fields."""
    from maseval.interface.agents.camel import CamelWorkforceTracer
    from unittest.mock import Mock

    # Create mock workforce
    mock_worker1 = Mock()
    mock_worker1.name = "worker_a"
    mock_worker2 = Mock()
    mock_worker2.name = "worker_b"

    mock_workforce = Mock()
    mock_workforce.mode = "collaborative"
    mock_workforce._children = [mock_worker1, mock_worker2]

    tracer = CamelWorkforceTracer(mock_workforce, name="test_workforce")

    config = tracer.gather_config()

    assert config["name"] == "test_workforce"
    assert config["mode"] == "collaborative"
    assert config["workers"] == ["worker_a", "worker_b"]


def test_workforce_tracer_extract_completed_tasks():
    """Test CamelWorkforceTracer extracts completed task info."""
    from maseval.interface.agents.camel import CamelWorkforceTracer
    from unittest.mock import Mock

    # Create mock tasks
    mock_task = Mock()
    mock_task.id = "task_123"
    mock_task.content = "Do something"
    mock_task.result = "Done"

    mock_workforce = Mock()
    mock_workforce._completed_tasks = [mock_task]
    mock_workforce._pending_tasks = []
    mock_workforce._assignees = {}
    mock_workforce._task_dependencies = {}

    tracer = CamelWorkforceTracer(mock_workforce)

    traces = tracer.gather_traces()

    assert len(traces["completed_tasks"]) == 1
    assert traces["completed_tasks"][0]["id"] == "task_123"
    assert traces["completed_tasks"][0]["content"] == "Do something"
    assert traces["completed_tasks"][0]["result"] == "Done"
