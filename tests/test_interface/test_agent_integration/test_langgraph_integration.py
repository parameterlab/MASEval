"""Integration tests for langgraph.

These tests require langgraph to be installed.
Run with: pytest -m langgraph
"""

import pytest

# Skip entire module if langgraph not installed
pytest.importorskip("langgraph")

# Mark all tests in this file as requiring langgraph
pytestmark = [pytest.mark.interface, pytest.mark.langgraph]


def test_langgraph_adapter_import():
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


def test_langgraph_adapter_message_manipulation():
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


def test_langgraph_adapter_message_manipulation_with_system_message():
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


def test_langgraph_adapter_logs_after_run():
    """Test that LangGraphAgentAdapter.logs is populated after run().

    This test validates that the manual logging implementation in LangGraphAgentAdapter
    captures all relevant execution information including:
    - Timing information (timestamp, duration)
    - Query information
    - Token usage (extracted from message metadata)
    - Status (success/error)
    - State information (keys, message count)
    - Checkpoint metadata (if available)
    """
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.messages.ai import UsageMetadata
    import time

    # Create a LangGraph agent with token usage metadata
    class State(TypedDict):
        messages: list

    def agent_node(state: State) -> State:
        messages = state["messages"]
        # Create AI message with usage metadata (simulates LLM response)
        # UsageMetadata is a TypedDict, so we create it properly
        response = AIMessage(
            content="Test response",
            usage_metadata=UsageMetadata(
                input_tokens=50,
                output_tokens=30,
                total_tokens=80,
            ),
        )
        return {"messages": messages + [response]}

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="test_agent")

    # Capture time before run
    time_before = time.time()

    # Run the agent
    result = adapter.run("Test query")

    # Capture time after run
    time_after = time.time()

    # Access logs
    logs = adapter.logs

    # Verify logs structure
    assert isinstance(logs, list)
    assert len(logs) >= 1  # At least one log entry

    # Get the most recent log entry
    log_entry = logs[-1]

    # Verify required fields
    assert "timestamp" in log_entry
    assert "query" in log_entry
    assert "duration_seconds" in log_entry
    assert "status" in log_entry

    # Verify field values
    assert log_entry["query"] == "Test query"
    assert log_entry["status"] == "success"
    assert log_entry["duration_seconds"] > 0
    assert log_entry["duration_seconds"] < (time_after - time_before) + 0.1  # Reasonable duration

    # Verify state information
    assert "state_keys" in log_entry
    assert "messages" in log_entry["state_keys"]
    assert "message_count" in log_entry
    assert log_entry["message_count"] >= 1

    # Verify token usage is captured from message metadata
    assert "input_tokens" in log_entry
    assert "output_tokens" in log_entry
    assert "total_tokens" in log_entry
    assert log_entry["input_tokens"] == 50
    assert log_entry["output_tokens"] == 30
    assert log_entry["total_tokens"] == 80


def test_langgraph_adapter_logs_multiple_runs():
    """Test that logs accumulate across multiple runs."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    from langchain_core.messages import AIMessage

    class State(TypedDict):
        messages: list

    def agent_node(state: State) -> State:
        messages = state["messages"]
        response = AIMessage(content="Response")
        return {"messages": messages + [response]}

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="test_agent")

    # First run
    adapter.run("Query 1")
    logs_after_first = adapter.logs
    assert len(logs_after_first) == 1
    assert logs_after_first[0]["query"] == "Query 1"

    # Second run
    adapter.run("Query 2")
    logs_after_second = adapter.logs
    assert len(logs_after_second) == 2
    assert logs_after_second[0]["query"] == "Query 1"
    assert logs_after_second[1]["query"] == "Query 2"


def test_langgraph_adapter_logs_error_handling():
    """Test that logs capture error information when agent execution fails."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict

    class State(TypedDict):
        messages: list

    def failing_node(state: State) -> State:
        raise ValueError("Intentional test error")

    graph = StateGraph(State)
    graph.add_node("agent", failing_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="test_agent")

    # Run should raise an error
    try:
        adapter.run("Test query")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Intentional test error" in str(e)

    # Verify error is logged
    logs = adapter.logs
    assert len(logs) == 1

    log_entry = logs[0]
    assert log_entry["status"] == "error"
    assert "error" in log_entry
    assert "error_type" in log_entry
    assert log_entry["error_type"] == "ValueError"
    assert "Intentional test error" in log_entry["error"]
    assert log_entry["query"] == "Test query"
    assert log_entry["duration_seconds"] >= 0


def test_langgraph_adapter_logs_without_token_metadata():
    """Test that logs work correctly when messages don't have usage metadata."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    from langchain_core.messages import AIMessage

    class State(TypedDict):
        messages: list

    def agent_node(state: State) -> State:
        messages = state["messages"]
        # Create response without usage metadata
        response = AIMessage(content="Test response")
        return {"messages": messages + [response]}

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="test_agent")
    adapter.run("Test query")

    # Verify logs exist but token fields are None or 0
    logs = adapter.logs
    assert len(logs) == 1

    log_entry = logs[0]
    # Token fields should be present but with default values
    assert log_entry.get("input_tokens") in [None, 0]
    assert log_entry.get("output_tokens") in [None, 0]
    assert log_entry.get("total_tokens") in [None, 0]
