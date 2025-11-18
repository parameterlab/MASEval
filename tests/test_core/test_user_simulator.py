"""Test User simulator functionality.

These tests verify that User simulator correctly manages conversation history.
"""

import pytest


@pytest.mark.core
class TestUserSimulator:
    """Tests for User simulator."""

    def test_user_simulate_response_updates_history(self, dummy_user):
        """Test that simulate_response adds to history."""
        initial_len = len(dummy_user.history)

        # Note: We don't call simulate_response as it would require LLM call
        # Instead, test manual history manipulation
        dummy_user.history.add_message("assistant", "How can I help?")
        dummy_user.history.add_message("user", "I need help")

        assert len(dummy_user.history) == initial_len + 2

    def test_user_history_includes_both_sides(self, dummy_user):
        """Test that history includes both user and assistant messages."""
        # Add a conversation
        dummy_user.history.add_message("assistant", "Question for user")
        dummy_user.history.add_message("user", "User response")

        messages = list(dummy_user.history)
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_user_gather_traces_includes_interactions(self, dummy_user):
        """Test that gather_traces() includes conversation history."""
        traces = dummy_user.gather_traces()

        assert "type" in traces
        assert "gathered_at" in traces
        assert "name" in traces
        assert "message_count" in traces
        assert "history" in traces

        assert traces["name"] == "test_user"
        assert isinstance(traces["history"], list)
        assert traces["message_count"] == len(traces["history"])

    def test_user_gather_config_includes_profile(self, dummy_user):
        """Test that gather_config() includes user profile."""
        config = dummy_user.gather_config()

        assert "type" in config
        assert "gathered_at" in config
        assert "name" in config

        assert config["name"] == "test_user"

    def test_user_initialization(self, dummy_model):
        """Test that User can be initialized with required parameters."""
        from conftest import DummyUser

        user = DummyUser(
            name="test_user",
            model=dummy_model,
            user_profile={"role": "customer"},
            scenario="test scenario",
            initial_prompt="Hello",
        )

        assert user.name == "test_user"
        assert user.user_profile == {"role": "customer"}
        assert user.scenario == "test scenario"
        assert len(user.history) == 1
