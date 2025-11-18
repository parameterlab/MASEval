"""Integration tests for langgraph.

These tests require langgraph to be installed.
Run with: pytest -m langgraph
"""

import pytest

# Skip entire module if langgraph not installed
pytest.importorskip("langgraph")

# Mark all tests in this file as requiring langgraph
pytestmark = [pytest.mark.interface, pytest.mark.langgraph]


def test_langgraph_wrapper_import():
    """Test that LangGraphAgentAdapter can be imported when langgraph is installed."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter, LangGraphUser

    assert LangGraphAgentAdapter is not None
    assert LangGraphUser is not None


def test_langgraph_in_agents_all():
    """Test that langgraph appears in interface.agents.__all__ when installed."""
    import maseval.interface.agents

    assert "LangGraphAgentAdapter" in maseval.interface.agents.__all__
    assert "LangGraphUser" in maseval.interface.agents.__all__


def test_check_langgraph_installed_function():
    """Test that _check_langgraph_installed doesn't raise when langgraph is installed."""
    from maseval.interface.agents.langgraph import _check_langgraph_installed

    # Should not raise
    _check_langgraph_installed()


def test_langgraph_wrapper_message_manipulation():
    """Test that LangGraphAgentAdapter supports message history manipulation.

    LangGraph supports manually managing message history through:
    - append_to_message_history: Add individual messages
    - set_message_history: Replace entire history
    - clear_message_history: Remove all messages
    - get_messages: Retrieve current history

    This is useful for multi-turn conversations and testing scenarios.
    """
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from maseval import MessageHistory
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    from langchain_core.messages import AIMessage

    # Create a simple LangGraph agent
    class State(TypedDict):
        messages: list

    def agent_node(state: State) -> State:
        messages = state["messages"]
        messages.append(AIMessage(content="Test response"))
        return {"messages": messages}

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    wrapper = LangGraphAgentAdapter(agent_instance=compiled, name="test_agent")

    # Test append_to_message_history
    wrapper.append_to_message_history("user", "First message")
    wrapper.append_to_message_history("assistant", "First response")

    history = wrapper.get_messages()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "First message"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "First response"

    # Test clear_message_history
    wrapper.clear_message_history()
    history = wrapper.get_messages()
    assert len(history) == 0

    # Test set_message_history
    new_history = MessageHistory()
    new_history.add_message("user", "Set message 1")
    new_history.add_message("assistant", "Set response 1")
    new_history.add_message("user", "Set message 2")

    wrapper.set_message_history(new_history)
    history = wrapper.get_messages()
    assert len(history) == 3
    assert history[0]["content"] == "Set message 1"
    assert history[1]["content"] == "Set response 1"
    assert history[2]["content"] == "Set message 2"

    # Verify history persists across multiple retrievals
    history_again = wrapper.get_messages()
    assert len(history_again) == 3
    assert history_again[0]["content"] == "Set message 1"


def test_langgraph_wrapper_message_manipulation_with_system_message():
    """Test message manipulation with system messages.

    Verifies that system messages are properly converted and handled
    when manipulating message history in LangGraph.
    """
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from maseval import MessageHistory
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    from langchain_core.messages import AIMessage

    # Create a simple LangGraph agent
    class State(TypedDict):
        messages: list

    def agent_node(state: State) -> State:
        messages = state["messages"]
        messages.append(AIMessage(content="Response"))
        return {"messages": messages}

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    wrapper = LangGraphAgentAdapter(agent_instance=compiled, name="test_agent")

    # Test set_message_history with system message
    new_history = MessageHistory()
    new_history.add_message("system", "You are a helpful assistant")
    new_history.add_message("user", "Hello")
    new_history.add_message("assistant", "Hi there")

    wrapper.set_message_history(new_history)
    history = wrapper.get_messages()

    assert len(history) == 3
    assert history[0]["role"] == "system"
    assert history[0]["content"] == "You are a helpful assistant"
    assert history[1]["role"] == "user"
    assert history[2]["role"] == "assistant"
