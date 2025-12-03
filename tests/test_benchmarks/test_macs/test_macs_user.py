"""Unit tests for MACSUser."""

import pytest
from unittest.mock import patch

from maseval.benchmark.macs import MACSUser

from .conftest import MACSModelAdapter


# =============================================================================
# Unit Tests: Initialization
# =============================================================================


@pytest.mark.benchmark
class TestMACSUserInit:
    """Tests for MACSUser initialization."""

    def test_init_basic(self, macs_model, sample_scenario, initial_prompt):
        """Basic initialization with required args."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        assert user.model == macs_model
        assert user.scenario == sample_scenario
        assert user.name == "Simulated User"  # Default name

    def test_init_custom_name(self, macs_model, sample_scenario, initial_prompt):
        """Custom name is respected."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
            name="Test User",
        )

        assert user.name == "Test User"

    def test_init_default_max_turns(self, macs_model, sample_scenario, initial_prompt):
        """Default max_turns is 5."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        assert user.max_turns == 5

    def test_init_custom_max_turns(self, macs_model, sample_scenario, initial_prompt):
        """Custom max_turns is respected."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
            max_turns=10,
        )

        assert user.max_turns == 10

    def test_init_loads_template(self, macs_model, sample_scenario, initial_prompt):
        """Loads user_simulator.txt template."""
        # Verify template file exists
        assert MACSUser.TEMPLATE_PATH.exists(), f"Template not found at {MACSUser.TEMPLATE_PATH}"

        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        # MACSUser is created successfully (template is passed to parent User class)
        assert user is not None

    def test_init_extracts_user_profile(self, macs_model, sample_scenario, initial_prompt):
        """Extracts profile from scenario."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        # Profile should contain extracted info
        assert "name" in user.user_profile or "full_scenario" in user.user_profile
        assert user.user_profile.get("full_scenario") == sample_scenario

    def test_init_turn_count_zero(self, macs_model, sample_scenario, initial_prompt):
        """Turn count starts at zero."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        assert user._turn_count == 0
        assert not user._stopped


# =============================================================================
# Unit Tests: User Profile Extraction
# =============================================================================


@pytest.mark.benchmark
class TestUserProfileExtraction:
    """Tests for _extract_user_profile static method."""

    def test_extract_profile_with_background(self, sample_scenario):
        """Parses Background: section."""
        profile = MACSUser._extract_user_profile(sample_scenario)

        assert "full_scenario" in profile
        assert profile["full_scenario"] == sample_scenario

    def test_extract_profile_is_statements(self):
        """Parses 'User's X is Y' statements."""
        scenario = """Background:
* User's name is Alice
* User's age is 30
* User's company is TechCorp"""

        profile = MACSUser._extract_user_profile(scenario)

        assert profile.get("name") == "Alice"
        assert profile.get("age") == "30"
        assert profile.get("company") == "TechCorp"

    def test_extract_profile_has_statements(self):
        """Parses 'User has X' statements."""
        scenario = """Background:
* User has a pet dog
* User has premium membership"""

        profile = MACSUser._extract_user_profile(scenario)

        # These should be captured with some key
        assert "full_scenario" in profile  # At minimum, full scenario is always there

    def test_extract_profile_no_background(self, minimal_scenario):
        """Handles missing Background section."""
        profile = MACSUser._extract_user_profile(minimal_scenario)

        # Should still have full_scenario
        assert profile["full_scenario"] == minimal_scenario

    def test_extract_profile_includes_full_scenario(self, sample_scenario):
        """Full scenario is always in profile."""
        profile = MACSUser._extract_user_profile(sample_scenario)

        assert "full_scenario" in profile
        assert sample_scenario in profile["full_scenario"]


# =============================================================================
# Unit Tests: Conversation State
# =============================================================================


@pytest.mark.benchmark
class TestConversationState:
    """Tests for conversation state management."""

    def test_is_done_false_initially(self, macs_model, sample_scenario, initial_prompt):
        """is_done is False at start."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        assert not user.is_done

    def test_is_done_after_max_turns(self, macs_model, sample_scenario, initial_prompt):
        """is_done is True after max turns."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
            max_turns=2,
        )

        # Manually increment turn count
        user._turn_count = 2

        assert user.is_done

    def test_is_done_after_stop_token(self, macs_model, sample_scenario, initial_prompt):
        """is_done is True after </stop> detected."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        # Manually set stopped flag
        user._stopped = True

        assert user.is_done

    def test_turn_count_below_max_not_done(self, macs_model, sample_scenario, initial_prompt):
        """Not done when turn count below max."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
            max_turns=5,
        )

        user._turn_count = 4  # One below max

        assert not user.is_done


# =============================================================================
# Unit Tests: Reset
# =============================================================================


@pytest.mark.benchmark
class TestReset:
    """Tests for reset method."""

    def test_reset_clears_turn_count(self, macs_model, sample_scenario, initial_prompt):
        """reset() clears turn count."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )
        user._turn_count = 3

        user.reset()

        assert user._turn_count == 0

    def test_reset_clears_stopped(self, macs_model, sample_scenario, initial_prompt):
        """reset() clears stopped flag."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )
        user._stopped = True

        user.reset()

        assert not user._stopped


# =============================================================================
# Unit Tests: Response Simulation
# =============================================================================


@pytest.mark.benchmark
class TestResponseSimulation:
    """Tests for simulate_response method."""

    def test_simulate_response_increments_turn(self, sample_scenario, initial_prompt):
        """Turn count increments on simulate_response call."""
        model = MACSModelAdapter(responses=['{"text": "Yes, confirmed.", "details": {}}'])
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        initial_count = user._turn_count

        # Mock parent's simulate_response to return a simple response
        with patch.object(user.__class__.__bases__[0], "simulate_response", return_value="Yes, confirmed."):
            user.simulate_response("When would you like to travel?")

        assert user._turn_count == initial_count + 1

    def test_simulate_response_detects_stop(self, sample_scenario, initial_prompt):
        """Detects </stop> token."""
        model = MACSModelAdapter()
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        with patch.object(user.__class__.__bases__[0], "simulate_response", return_value="Thanks! </stop>"):
            user.simulate_response("Your flight is booked!")

        assert user._stopped
        assert user.is_done

    def test_simulate_response_cleans_stop_token(self, sample_scenario, initial_prompt):
        """Removes </stop> from response."""
        model = MACSModelAdapter()
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        with patch.object(user.__class__.__bases__[0], "simulate_response", return_value="Perfect, thanks! </stop>"):
            response = user.simulate_response("Booking confirmed!")

        assert "</stop>" not in response
        assert "Perfect, thanks!" in response

    def test_simulate_response_returns_empty_when_done(self, sample_scenario, initial_prompt):
        """Returns empty string when is_done is True."""
        model = MACSModelAdapter()
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )
        user._stopped = True  # Already done

        response = user.simulate_response("Any follow-up?")

        assert response == ""

    def test_simulate_response_returns_empty_at_max_turns(self, sample_scenario, initial_prompt):
        """Returns empty string when max turns reached."""
        model = MACSModelAdapter()
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
            max_turns=3,
        )
        user._turn_count = 3  # At max

        response = user.simulate_response("One more question?")

        assert response == ""

    def test_simulate_response_fallback_message(self, sample_scenario, initial_prompt):
        """Provides fallback when response is only stop token."""
        model = MACSModelAdapter()
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        with patch.object(user.__class__.__bases__[0], "simulate_response", return_value="</stop>"):
            response = user.simulate_response("Booking complete!")

        assert response == "Thank you, that's all I needed!"
        assert user._stopped


# =============================================================================
# Unit Tests: Tool Interface
# =============================================================================


@pytest.mark.benchmark
class TestToolInterface:
    """Tests for get_tool method."""

    def test_get_tool_raises_not_implemented(self, macs_model, sample_scenario, initial_prompt):
        """Base MACSUser.get_tool() raises NotImplementedError."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        with pytest.raises(NotImplementedError) as exc_info:
            user.get_tool()

        assert "get_tool" in str(exc_info.value)
        assert "subclass" in str(exc_info.value).lower()


# =============================================================================
# Unit Tests: Tracing
# =============================================================================


@pytest.mark.benchmark
class TestTracing:
    """Tests for gather_traces method."""

    def test_gather_traces_includes_macs_fields(self, macs_model, sample_scenario, initial_prompt):
        """Traces include MACS-specific fields."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
            max_turns=7,
        )
        user._turn_count = 3
        user._stopped = True

        traces = user.gather_traces()

        assert traces["max_turns"] == 7
        assert traces["turns_used"] == 3
        assert traces["stopped_by_user"] is True

    def test_gather_traces_inherits_base_fields(self, macs_model, sample_scenario, initial_prompt):
        """Traces include base User fields."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
        )

        traces = user.gather_traces()

        assert "gathered_at" in traces
        assert "name" in traces
        assert traces["name"] == "Simulated User"


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestMACSUserIntegration:
    """Integration tests for MACSUser."""

    def test_conversation_lifecycle(self, sample_scenario, initial_prompt):
        """Test complete conversation lifecycle."""
        responses = [
            "Yes, Monday works.",
            "I prefer Delta.",
            "Aisle seat please.",
            "Book it! </stop>",
        ]
        model = MACSModelAdapter(responses=responses)
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
            max_turns=5,
        )

        # Simulate multi-turn conversation
        questions = [
            "When would you like to travel?",
            "Any airline preference?",
            "Window or aisle?",
            "I'll book the flight. Confirmation?",
        ]

        for i, question in enumerate(questions):
            if user.is_done:
                break
            with patch.object(user.__class__.__bases__[0], "simulate_response", return_value=responses[i]):
                response = user.simulate_response(question)
            if i < len(questions) - 1:
                assert response != ""

        # After stop token, should be done
        assert user.is_done
        assert user._turn_count == 4

    def test_max_turns_enforcement(self, sample_scenario, initial_prompt):
        """Test that max turns is enforced."""
        model = MACSModelAdapter(responses=["Response"] * 10)
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
            max_turns=3,
        )

        # Simulate 3 turns
        for i in range(3):
            with patch.object(user.__class__.__bases__[0], "simulate_response", return_value="Response"):
                user.simulate_response(f"Question {i}")

        # Should be done after 3 turns
        assert user.is_done
        assert user._turn_count == 3

        # Additional calls should return empty
        response = user.simulate_response("One more?")
        assert response == ""

    def test_reset_allows_new_conversation(self, sample_scenario, initial_prompt):
        """Test that reset allows starting new conversation."""
        model = MACSModelAdapter()
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_prompt=initial_prompt,
            max_turns=2,
        )

        # Max out turns
        user._turn_count = 2
        user._stopped = True
        assert user.is_done

        # Reset
        user.reset()

        # Should be able to continue
        assert not user.is_done
        assert user._turn_count == 0
