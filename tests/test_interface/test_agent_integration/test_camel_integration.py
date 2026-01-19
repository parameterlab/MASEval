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
