"""Test User class functionality.

These tests verify the User base class (maseval.core.user.User) behavior:
- Conversation history management (MessageHistory)
- Multi-turn interaction (max_turns, turn counting)
- Early stopping via stop tokens
- Optional initial prompts and LLM-generated initial queries

Note: This tests the User class, NOT the UserLLMSimulator class.
UserLLMSimulator is tested in test_llm_simulator.py.
"""

import pytest


@pytest.mark.core
class TestUser:
    """Tests for User base class basics."""

    def test_user_simulate_response_updates_messages(self, dummy_user):
        """Test that simulate_response adds to message history."""
        initial_len = len(dummy_user.messages)

        # simulate_response adds assistant message, then user response
        dummy_user.simulate_response("How can I help?")

        # Should have added 2 messages: assistant question + user response
        assert len(dummy_user.messages) == initial_len + 2

    def test_user_messages_includes_both_sides(self, dummy_user):
        """Test that messages includes both user and assistant messages."""
        # Simulate a response (adds assistant + user messages)
        dummy_user.simulate_response("Question for user")

        messages = list(dummy_user.messages)
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
        assert "messages" in traces

        assert traces["name"] == "test_user"
        assert isinstance(traces["messages"], list)
        assert traces["message_count"] == len(traces["messages"])

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
        assert len(user.messages) == 1


# =============================================================================
# Multi-Turn Configuration Tests
# =============================================================================


@pytest.mark.core
class TestUserMultiTurn:
    """Tests for max_turns behavior."""

    def test_default_max_turns_is_one(self, dummy_model):
        """Default single-turn mode."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        assert user.max_turns == 1

    def test_custom_max_turns(self, dummy_model):
        """Custom max_turns is stored correctly."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=5)
        assert user.max_turns == 5

    def test_is_done_after_max_turns(self, dummy_model):
        """is_done() returns True when turn count >= max_turns."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=2)
        user._turn_count = 2

        assert user.is_done()

    def test_is_done_before_max_turns(self, dummy_model):
        """is_done() returns False when turn count < max_turns."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=3)
        user._turn_count = 1

        assert not user.is_done()

    def test_simulate_response_increments_turn_count(self, dummy_model):
        """Each simulate_response() call increments _turn_count."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=5)
        initial_count = user._turn_count

        user.simulate_response("Question 1")
        assert user._turn_count == initial_count + 1

        user.simulate_response("Question 2")
        assert user._turn_count == initial_count + 2

    def test_simulate_response_returns_empty_when_done(self, dummy_model):
        """Returns empty string when is_done() is True."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=1)
        user._turn_count = 1  # Already at max

        response = user.simulate_response("More questions?")
        assert response == ""

    def test_turn_count_starts_at_zero(self, dummy_model):
        """Turn count starts at 0."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        assert user._turn_count == 0


# =============================================================================
# Stop Token Tests
# =============================================================================


@pytest.mark.core
class TestUserStopToken:
    """Tests for stop_token early termination."""

    def test_no_stop_token_by_default(self, dummy_model):
        """stop_token is None by default."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        assert user.stop_token is None

    def test_custom_stop_token(self, dummy_model):
        """Custom stop_token is stored."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, stop_token="</done>")
        assert user.stop_token == "</done>"

    def test_stop_token_detection_sets_stopped(self, dummy_model):
        """Detecting stop token sets _stopped = True."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, stop_token="</stop>", max_turns=5)
        user.simulator.return_value = "Thanks! </stop>"

        user.simulate_response("Here's your answer")

        assert user._stopped

    def test_stop_token_removed_from_response(self, dummy_model):
        """Stop token is stripped from returned response."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, stop_token="</stop>", max_turns=5)
        user.simulator.return_value = "Perfect, thanks! </stop>"

        response = user.simulate_response("Booking confirmed!")

        assert "</stop>" not in response
        assert "Perfect, thanks!" in response

    def test_is_done_true_after_stop_token(self, dummy_model):
        """is_done() returns True after stop token detected."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, stop_token="</stop>", max_turns=5)
        user.simulator.return_value = "Done </stop>"

        user.simulate_response("Result")

        assert user.is_done()

    def test_stop_token_case_insensitive(self, dummy_model):
        """Stop token detection is case-insensitive."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, stop_token="</STOP>", max_turns=5)
        user.simulator.return_value = "Thanks! </stop>"  # lowercase

        user.simulate_response("Answer")

        assert user._stopped

    def test_fallback_message_when_only_stop_token(self, dummy_model):
        """Provides fallback when response is only stop token."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, stop_token="</stop>", max_turns=5)
        user.simulator.return_value = "</stop>"

        response = user.simulate_response("Done!")

        assert response == "Thank you, that's all I needed!"
        assert user._stopped

    def test_stop_token_response_counts_as_turn(self, dummy_model):
        """The response containing stop token still counts as a turn."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, stop_token="</stop>", max_turns=5)
        user.simulator.return_value = "Thank you, all is clear </stop>"

        initial_turn_count = user._turn_count
        user.simulate_response("Here is your result")

        # Turn count should increment even though stop token was detected
        assert user._turn_count == initial_turn_count + 1
        assert user._stopped
        assert user.is_done()


# =============================================================================
# Optional Initial Prompt Tests
# =============================================================================


@pytest.mark.core
class TestUserInitialPrompt:
    """Tests for optional initial_prompt behavior."""

    def test_with_initial_prompt_adds_message(self, dummy_model):
        """Providing initial_prompt adds it to messages."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="I need help booking a flight",
        )

        assert len(user.messages) == 1
        assert user.messages[0]["role"] == "user"
        assert user.messages[0]["content"] == "I need help booking a flight"

    def test_without_initial_prompt_empty_messages(self, dummy_model):
        """No initial_prompt means empty message history."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        # No initial_prompt provided

        assert len(user.messages) == 0

    def test_get_initial_query_generates_message(self, dummy_model):
        """get_initial_query() uses LLM to generate first message."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        user.simulator.return_value = "I want to book a hotel"

        query = user.get_initial_query()

        assert query == "I want to book a hotel"
        user.simulator.assert_called_once()

    def test_get_initial_query_adds_to_messages(self, dummy_model):
        """Generated query is added to message history."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        user.simulator.return_value = "Help me please"

        user.get_initial_query()

        assert len(user.messages) == 1
        assert user.messages[0]["role"] == "user"
        assert user.messages[0]["content"] == "Help me please"

    def test_get_initial_query_raises_if_messages_exist(self, dummy_model):
        """get_initial_query() raises if messages already exist."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="Already have a message",
        )

        with pytest.raises(RuntimeError, match="already has messages"):
            user.get_initial_query()

    def test_get_initial_query_not_counted_as_turn(self, dummy_model):
        """Initial query doesn't increment turn count."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=3)
        user.simulator.return_value = "Initial query"

        user.get_initial_query()

        assert user._turn_count == 0  # Not incremented


# =============================================================================
# Message History Completeness Tests
# =============================================================================


@pytest.mark.core
class TestUserMessageHistory:
    """Tests for complete message tracing."""

    def test_initial_message_in_history(self, dummy_model):
        """Initial prompt is in message history."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="Hello agent",
        )

        assert len(user.messages) == 1
        assert user.messages[0]["content"] == "Hello agent"

    def test_assistant_message_recorded(self, dummy_model):
        """simulate_response() records assistant message before responding."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=3)
        user.simulator.return_value = "User reply"

        user.simulate_response("Agent says hello")

        # Should have: assistant message + user response
        assert len(user.messages) == 2
        assert user.messages[0]["role"] == "assistant"
        assert user.messages[0]["content"] == "Agent says hello"

    def test_user_response_recorded(self, dummy_model):
        """simulate_response() records user response."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=3)
        user.simulator.return_value = "Thanks for the help"

        user.simulate_response("Here's your answer")

        assert user.messages[-1]["role"] == "user"
        assert user.messages[-1]["content"] == "Thanks for the help"

    def test_full_conversation_tracked(self, dummy_model):
        """Multiple exchanges create complete conversation trace."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="I need a flight",
            max_turns=3,
        )
        user.simulator.side_effect = ["Monday works", "Yes, book it"]

        # Two agent-user exchanges
        user.simulate_response("When do you want to travel?")
        user.simulate_response("Shall I book it?")

        messages = list(user.messages)
        assert len(messages) == 5  # initial + 2*(assistant + user)

        # Verify order
        assert messages[0]["role"] == "user"  # initial
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

    def test_gather_traces_includes_all_messages(self, dummy_model):
        """gather_traces() includes complete conversation."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="Hello",
            max_turns=2,
        )
        user.simulator.return_value = "Got it"

        user.simulate_response("Agent response")

        traces = user.gather_traces()

        assert traces["message_count"] == 3
        assert len(traces["messages"]) == 3


# =============================================================================
# Config Tests
# =============================================================================


@pytest.mark.core
class TestUserConfig:
    """Tests for gather_config updates."""

    def test_config_includes_max_turns(self, dummy_model):
        """gather_config() includes max_turns."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=7)

        config = user.gather_config()

        assert config["max_turns"] == 7

    def test_config_includes_stop_token(self, dummy_model):
        """gather_config() includes stop_token."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, stop_token="</end>")

        config = user.gather_config()

        assert config["stop_token"] == "</end>"

    def test_config_includes_none_stop_token(self, dummy_model):
        """gather_config() includes stop_token even when None."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)

        config = user.gather_config()

        assert "stop_token" in config
        assert config["stop_token"] is None
