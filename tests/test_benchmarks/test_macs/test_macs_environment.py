"""Unit tests for MACSEnvironment."""

import pytest
from unittest.mock import patch

from maseval.benchmark.macs import MACSEnvironment, MACSGenericTool

from .conftest import MACSModelAdapter


# =============================================================================
# Unit Tests: Initialization
# =============================================================================


@pytest.mark.benchmark
class TestMACSEnvironmentInit:
    """Tests for MACSEnvironment initialization."""

    def test_init_with_task_data(self, macs_model, sample_task_data):
        """Initializes from task data."""
        env = MACSEnvironment(sample_task_data, macs_model)

        assert env is not None
        assert "tool_specs" in env.state

    def test_init_stores_model(self, macs_model, sample_task_data):
        """Model is stored for tool creation."""
        env = MACSEnvironment(sample_task_data, macs_model)

        assert env._model == macs_model

    def test_init_calls_parent(self, macs_model, sample_task_data):
        """Parent Environment.__init__ is called."""
        env = MACSEnvironment(sample_task_data, macs_model)

        # Parent sets up state and creates tools
        assert hasattr(env, "state")
        assert hasattr(env, "tools")


# =============================================================================
# Unit Tests: State Setup
# =============================================================================


@pytest.mark.benchmark
class TestSetupState:
    """Tests for setup_state method."""

    def test_setup_state_extracts_tool_specs(self, macs_model, sample_task_data):
        """setup_state extracts tool_specs from task_data."""
        env = MACSEnvironment(sample_task_data, macs_model)

        assert "tool_specs" in env.state
        assert len(env.state["tool_specs"]) == 2

    def test_setup_state_empty_tools(self, macs_model):
        """Handles missing or empty tools."""
        task_data = {"environment_data": {}}
        env = MACSEnvironment(task_data, macs_model)

        assert env.state["tool_specs"] == []

    def test_setup_state_no_environment_data(self, macs_model):
        """Handles missing environment_data."""
        task_data = {}
        env = MACSEnvironment(task_data, macs_model)

        assert env.state["tool_specs"] == []


# =============================================================================
# Unit Tests: Tool Creation
# =============================================================================


@pytest.mark.benchmark
class TestCreateTools:
    """Tests for create_tools method."""

    def test_create_tools_from_specs(self, macs_model, sample_task_data):
        """Creates MACSGenericTool instances from specs."""
        env = MACSEnvironment(sample_task_data, macs_model)

        assert len(env.tools) == 3  # search_flights, book_flight, search_hotels
        assert all(isinstance(tool, MACSGenericTool) for tool in env.tools.values())

    def test_create_tools_keyed_by_name(self, macs_model, sample_task_data):
        """Tools dict is keyed by tool name."""
        env = MACSEnvironment(sample_task_data, macs_model)

        assert "search_flights" in env.tools
        assert "book_flight" in env.tools
        assert "search_hotels" in env.tools

    def test_create_tools_correct_properties(self, macs_model, sample_task_data):
        """Created tools have correct properties."""
        env = MACSEnvironment(sample_task_data, macs_model)

        search_flights = env.tools["search_flights"]
        assert search_flights.name == "search_flights"
        assert search_flights.description == "Search for available flights"
        assert "origin" in search_flights.inputs
        assert "destination" in search_flights.inputs

    def test_create_tools_deduplicates(self, macs_model):
        """Duplicate tool names are deduplicated."""
        task_data = {
            "environment_data": {
                "tools": [
                    {
                        "tool_name": "group1",
                        "actions": [{"name": "duplicate_tool", "description": "First"}],
                    },
                    {
                        "tool_name": "group2",
                        "actions": [{"name": "duplicate_tool", "description": "First"}],  # Same name
                    },
                ]
            }
        }
        env = MACSEnvironment(task_data, macs_model)

        # Should only have one instance
        assert len(env.tools) == 1
        assert "duplicate_tool" in env.tools

    def test_create_tools_empty_specs(self, macs_model):
        """Empty specs returns empty dict."""
        task_data = {"environment_data": {"tools": []}}
        env = MACSEnvironment(task_data, macs_model)

        assert env.tools == {}

    def test_create_tools_empty_actions(self, macs_model):
        """Handles tool groups with no actions."""
        task_data = {
            "environment_data": {
                "tools": [
                    {"tool_name": "empty_group", "actions": []},
                ]
            }
        }
        env = MACSEnvironment(task_data, macs_model)

        assert env.tools == {}


# =============================================================================
# Unit Tests: Agent Tool Assignment
# =============================================================================


@pytest.mark.benchmark
class TestGetToolsForAgent:
    """Tests for get_tools_for_agent method."""

    def test_get_tools_for_agent(self, macs_model, sample_task_data, sample_agent_spec_flight):
        """Returns tools matching agent spec."""
        env = MACSEnvironment(sample_task_data, macs_model)

        agent_tools = env.get_tools_for_agent(sample_agent_spec_flight)

        assert len(agent_tools) == 2  # search_flights, book_flight
        assert "search_flights" in agent_tools
        assert "book_flight" in agent_tools
        assert "search_hotels" not in agent_tools

    def test_get_tools_for_agent_all(self, macs_model, sample_task_data, sample_agent_spec_all):
        """Returns all tools when agent has access to all groups."""
        env = MACSEnvironment(sample_task_data, macs_model)

        agent_tools = env.get_tools_for_agent(sample_agent_spec_all)

        assert len(agent_tools) == 3
        assert "search_flights" in agent_tools
        assert "book_flight" in agent_tools
        assert "search_hotels" in agent_tools

    def test_get_tools_for_agent_no_match(self, macs_model, sample_task_data, sample_agent_spec_none):
        """Returns empty dict if no matching tool groups."""
        env = MACSEnvironment(sample_task_data, macs_model)

        agent_tools = env.get_tools_for_agent(sample_agent_spec_none)

        assert agent_tools == {}

    def test_get_tools_for_agent_partial(self, macs_model, sample_task_data):
        """Returns subset matching agent's tool groups."""
        env = MACSEnvironment(sample_task_data, macs_model)

        agent_spec = {
            "agent_id": "hotel_agent",
            "tools": ["hotel_tools"],
        }
        agent_tools = env.get_tools_for_agent(agent_spec)

        assert len(agent_tools) == 1
        assert "search_hotels" in agent_tools

    def test_get_tools_for_agent_returns_same_instances(self, macs_model, sample_task_data, sample_agent_spec_flight):
        """Returns same tool instances as in env.tools."""
        env = MACSEnvironment(sample_task_data, macs_model)

        agent_tools = env.get_tools_for_agent(sample_agent_spec_flight)

        # Same instance, not copies
        assert agent_tools["search_flights"] is env.tools["search_flights"]

    def test_get_tools_for_agent_empty_tools_list(self, macs_model, sample_task_data):
        """Handles agent with empty tools list."""
        env = MACSEnvironment(sample_task_data, macs_model)

        agent_spec = {"agent_id": "no_tools", "tools": []}
        agent_tools = env.get_tools_for_agent(agent_spec)

        assert agent_tools == {}


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestMACSEnvironmentIntegration:
    """Integration tests for MACSEnvironment."""

    def test_full_workflow(self, macs_model, sample_task_data):
        """Test complete environment workflow."""
        # Create environment
        env = MACSEnvironment(sample_task_data, macs_model)

        # Verify tools created
        assert len(env.tools) == 3

        # Get tools for different agents
        flight_agent_spec = {"agent_id": "flight", "tools": ["flight_tools"]}
        hotel_agent_spec = {"agent_id": "hotel", "tools": ["hotel_tools"]}
        supervisor_spec = {"agent_id": "super", "tools": ["flight_tools", "hotel_tools"]}

        flight_tools = env.get_tools_for_agent(flight_agent_spec)
        hotel_tools = env.get_tools_for_agent(hotel_agent_spec)
        supervisor_tools = env.get_tools_for_agent(supervisor_spec)

        assert len(flight_tools) == 2
        assert len(hotel_tools) == 1
        assert len(supervisor_tools) == 3

    def test_tools_are_callable(self, sample_task_data):
        """Created tools can be called."""
        # Use a model that returns valid JSON responses (ToolLLMSimulator expects {"text": ..., "details": ...})
        model = MACSModelAdapter(responses=['{"text": "Found flights: AA123, UA456", "details": {}}'])
        env = MACSEnvironment(sample_task_data, model)

        search_flights = env.tools["search_flights"]
        result = search_flights(origin="LAX", destination="JFK")

        # Should return the text from the response
        assert "Found flights" in result

    def test_multiple_agents_share_tools(self, macs_model, sample_task_data):
        """Multiple agents can share the same tool instances."""
        env = MACSEnvironment(sample_task_data, macs_model)

        agent1_spec = {"agent_id": "agent1", "tools": ["flight_tools"]}
        agent2_spec = {"agent_id": "agent2", "tools": ["flight_tools"]}

        agent1_tools = env.get_tools_for_agent(agent1_spec)
        agent2_tools = env.get_tools_for_agent(agent2_spec)

        # Same tool instances
        assert agent1_tools["search_flights"] is agent2_tools["search_flights"]

        # Invocation history is shared
        with patch.object(agent1_tools["search_flights"].simulator, "__call__", return_value=("Result", {})):
            agent1_tools["search_flights"](origin="LAX", destination="JFK")

        # Agent2's tool (same instance) should have the invocation
        assert len(agent2_tools["search_flights"].history.to_list()) == 1


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.benchmark
class TestEdgeCases:
    """Edge case tests for MACSEnvironment."""

    def test_tool_with_no_name(self, macs_model):
        """Handles actions without name field."""
        task_data = {
            "environment_data": {
                "tools": [
                    {
                        "tool_name": "group",
                        "actions": [
                            {"description": "No name field"},  # Missing name
                            {"name": "valid_tool", "description": "Has name"},
                        ],
                    }
                ]
            }
        }
        env = MACSEnvironment(task_data, macs_model)

        # Should only create the valid tool
        assert len(env.tools) == 1
        assert "valid_tool" in env.tools

    def test_callbacks_passed_to_parent(self, macs_model, sample_task_data):
        """Callbacks are passed to parent Environment."""
        from maseval.core.callback import EnvironmentCallback

        # Create actual callback instances
        class MockCallback(EnvironmentCallback):
            pass

        callbacks = [MockCallback(), MockCallback()]
        env = MACSEnvironment(sample_task_data, macs_model, callbacks=callbacks)

        assert len(env.callbacks) == 2
        assert all(isinstance(cb, EnvironmentCallback) for cb in env.callbacks)

    def test_nested_tool_groups(self, macs_model):
        """Handles deeply nested tool structures."""
        task_data = {
            "environment_data": {
                "tools": [
                    {
                        "tool_name": "level1",
                        "actions": [
                            {
                                "name": "tool1",
                                "description": "Tool 1",
                                "input_schema": {
                                    "properties": {
                                        "nested": {
                                            "type": "object",
                                            "description": "Nested object",
                                        }
                                    }
                                },
                            }
                        ],
                    }
                ]
            }
        }
        env = MACSEnvironment(task_data, macs_model)

        assert "tool1" in env.tools
        assert "nested" in env.tools["tool1"].inputs
