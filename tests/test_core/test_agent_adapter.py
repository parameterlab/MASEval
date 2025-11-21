"""Test AgentAdapter functionality.

These tests verify that AgentAdapter provides the correct interface for
adapting agents from any framework.
"""

import pytest
from maseval import MessageHistory


@pytest.mark.core
class TestAgentAdapter:
    """Tests for AgentAdapter interface and behavior."""

    def test_agent_adapter_run_triggers_callbacks(self, dummy_agent_adapter):
        """Test that run() triggers agent callbacks."""
        from maseval import AgentCallback

        # Track callback invocations
        callback_calls = []

        class TrackingCallback(AgentCallback):
            def on_run_start(self, agent):
                callback_calls.append(("start", agent.name))

            def on_run_end(self, agent, result):
                callback_calls.append(("end", agent.name, result))

        dummy_agent_adapter.callbacks = [TrackingCallback()]
        _ = dummy_agent_adapter.run("Test query")

        assert len(callback_calls) == 2
        assert callback_calls[0] == ("start", "test_agent")
        assert callback_calls[1][0] == "end"
        assert callback_calls[1][1] == "test_agent"
        assert "Response to: Test query" in callback_calls[1][2]

    def test_agent_adapter_get_messages_returns_history(self, dummy_agent_adapter):
        """Test that get_messages() returns MessageHistory."""
        # Before run, should return empty history
        history = dummy_agent_adapter.get_messages()
        assert isinstance(history, MessageHistory)
        assert len(history) == 0

        # After run, should have messages
        dummy_agent_adapter.run("Test query")
        history = dummy_agent_adapter.get_messages()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_agent_adapter_gather_traces_includes_messages(self, dummy_agent_adapter):
        """Test that gather_traces() includes message history."""
        dummy_agent_adapter.run("Test query")

        traces = dummy_agent_adapter.gather_traces()

        assert "type" in traces
        assert "gathered_at" in traces
        assert "name" in traces
        assert "agent_type" in traces
        assert "message_count" in traces
        assert "messages" in traces

        assert traces["name"] == "test_agent"
        assert traces["message_count"] == 2
        assert len(traces["messages"]) == 2

    def test_agent_adapter_gather_config(self, dummy_agent_adapter):
        """Test that gather_config() returns configuration."""
        config = dummy_agent_adapter.gather_config()

        assert "type" in config
        assert "gathered_at" in config
        assert "name" in config
        assert "agent_type" in config

        assert config["name"] == "test_agent"
        assert config["type"] == "DummyAgentAdapter"

    def test_agent_adapter_multiple_runs(self, dummy_agent_adapter):
        """Test that adapter can be run multiple times and history accumulates."""
        result1 = dummy_agent_adapter.run("Query 1")
        assert "Query 1" in result1

        result2 = dummy_agent_adapter.run("Query 2")
        assert "Query 2" in result2

        # History should have both runs
        history = dummy_agent_adapter.get_messages()
        assert len(history) == 4  # 2 messages per run
        assert history[0]["content"] == "Query 1"
        assert history[2]["content"] == "Query 2"
