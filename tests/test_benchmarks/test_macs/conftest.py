"""Shared fixtures for MACS benchmark tests.

Fixture Hierarchy
-----------------
- tests/conftest.py: Generic fixtures (dummy_model, dummy_agent_adapter, dummy_task, etc.)
  These are automatically available via pytest's conftest inheritance.
- tests/test_benchmarks/test_macs/conftest.py: MACS-specific fixtures (this file)

MACS tests can use fixtures from both levels - pytest handles this automatically.

Why MACS-Specific Mock Classes Exist
------------------------------------
The MACS benchmark uses ToolLLMSimulator and UserLLMSimulator which parse JSON responses
in a specific format: {"text": "...", "details": {...}}

The generic DummyModelAdapter from tests/conftest.py returns simple strings like
"test response", which would cause JSON parsing failures in MACS components.

Therefore, we define MACS-specific adapters that:
1. MACSModelAdapter: Returns valid JSON in the ToolLLMSimulator format by default
2. MACSAgentAdapter: Returns MessageHistory (not strings) matching the AgentAdapter contract

These are NOT duplicates - they serve a different purpose than the generic test fixtures.
"""

import pytest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

from maseval import AgentAdapter, Task, User, MessageHistory, TaskCollection
from maseval.benchmark.macs import MACSBenchmark, MACSEnvironment
from maseval.core.model import ModelAdapter


# =============================================================================
# MACS-Specific Mock Components
#
# These exist because MACS components (ToolLLMSimulator, UserLLMSimulator, MACSEvaluator)
# expect JSON responses in specific formats. The generic DummyModelAdapter returns
# plain strings which would cause parsing failures.
# =============================================================================


class MACSModelAdapter(ModelAdapter):
    """Model adapter for testing MACS components.

    Unlike DummyModelAdapter (which returns plain strings), this adapter returns
    JSON responses in the format expected by MACS simulators:

        {"text": "response text", "details": {...}}

    This format is required by:
    - ToolLLMSimulator._parse_output() for tool responses
    - UserLLMSimulator._parse_output() for user responses
    - MACSEvaluator for assertion evaluation (different format)

    Attributes:
        prompts: List of all prompts sent to the model (for verification in tests).
        _call_count: Number of times generate() was called.
    """

    def __init__(self, responses: Optional[List[str]] = None):
        """Initialize with optional canned responses.

        Args:
            responses: List of JSON strings to return. Cycles through if more
                calls are made than responses provided. Defaults to a valid
                ToolLLMSimulator response format.
        """
        super().__init__()
        self._model_id = "macs-test-model"
        self._responses = responses or ['{"text": "Default response", "details": {}}']
        self._call_count = 0
        self.prompts: List[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def _generate_impl(self, prompt: str, generation_params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return response


class MACSAgentAdapter(AgentAdapter):
    """Agent adapter for testing MACS benchmark execution.

    Unlike DummyAgentAdapter (which wraps a real agent object), this adapter
    provides controllable responses without needing a real agent implementation.

    Used for testing MACSBenchmark.run_agents() and integration scenarios.

    Attributes:
        run_calls: List of queries passed to _run_agent (for verification).
    """

    def __init__(self, name: str = "macs_test_agent"):
        super().__init__(agent_instance=MagicMock(), name=name)
        self._responses: List[str] = []
        self._call_count = 0
        self.run_calls: List[str] = []

    def set_responses(self, responses: List[str]) -> None:
        """Set canned responses for the agent."""
        self._responses = responses

    def _run_agent(self, query: str) -> MessageHistory:
        self.run_calls.append(query)
        if self._responses:
            response = self._responses[self._call_count % len(self._responses)]
            self._call_count += 1
        else:
            response = f"Response to: {query}"
        return MessageHistory([{"role": "assistant", "content": response}])


# =============================================================================
# MACS-Specific Benchmark Implementation
# =============================================================================


class ConcreteMACSBenchmark(MACSBenchmark):
    """Concrete MACSBenchmark implementation for testing.

    MACSBenchmark is abstract (setup_agents must be implemented by users).
    This provides a minimal implementation using MACSAgentAdapter.
    """

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: MACSEnvironment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create test agents using MACSAgentAdapter."""
        adapter = MACSAgentAdapter("macs_test_agent")
        return [adapter], {"macs_test_agent": adapter}


# =============================================================================
# Model Fixtures
#
# These use MACSModelAdapter because MACS components require JSON responses.
# For generic model testing, use dummy_model from parent conftest.
# =============================================================================


@pytest.fixture
def macs_model():
    """MACS model adapter with default JSON responses.

    Returns responses in ToolLLMSimulator format: {"text": "...", "details": {...}}
    """
    return MACSModelAdapter()


@pytest.fixture
def macs_model_evaluator():
    """MACS model configured for MACSEvaluator tests.

    Returns JSON array format expected by MACSEvaluator._parse_evaluation_response().
    """
    return MACSModelAdapter(responses=['[{"assertion": "Test", "answer": "TRUE", "evidence": "OK"}]'])


@pytest.fixture
def macs_model_tool():
    """MACS model configured for ToolLLMSimulator tests."""
    return MACSModelAdapter(responses=['{"text": "Tool executed successfully", "details": {}}'])


@pytest.fixture
def macs_model_user():
    """MACS model configured for UserLLMSimulator tests."""
    return MACSModelAdapter(responses=['{"text": "Yes, that works for me.", "details": {}}'])


# =============================================================================
# MACS Tool Specification Fixtures
# =============================================================================


@pytest.fixture
def simple_tool_spec():
    """Simple tool specification for basic tests."""
    return {
        "name": "search_flights",
        "description": "Search for available flights",
        "input_schema": {
            "properties": {
                "origin": {"type": "string", "description": "Origin airport code"},
                "destination": {"type": "string", "description": "Destination airport code"},
            }
        },
    }


@pytest.fixture
def complex_tool_spec():
    """Tool specification with various input types."""
    return {
        "name": "book_hotel",
        "description": "Book a hotel room",
        "input_schema": {
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "check_in": {"data_type": "date", "description": "Check-in date"},
                "guests": {"type": "integer", "description": "Number of guests"},
                "amenities": {"type": "array", "description": "Requested amenities"},
            }
        },
    }


@pytest.fixture
def minimal_tool_spec():
    """Minimal tool specification with only name."""
    return {"name": "simple_action"}


@pytest.fixture
def sample_tool_specs():
    """Sample tool specifications in MACS format (tool groups with actions)."""
    return [
        {
            "tool_name": "flight_tools",
            "actions": [
                {
                    "name": "search_flights",
                    "description": "Search for available flights",
                    "input_schema": {
                        "properties": {
                            "origin": {"type": "string", "description": "Origin airport"},
                            "destination": {"type": "string", "description": "Destination airport"},
                        }
                    },
                },
                {
                    "name": "book_flight",
                    "description": "Book a flight",
                    "input_schema": {
                        "properties": {
                            "flight_id": {"type": "string", "description": "Flight ID to book"},
                        }
                    },
                },
            ],
        },
        {
            "tool_name": "hotel_tools",
            "actions": [
                {
                    "name": "search_hotels",
                    "description": "Search for hotels",
                    "input_schema": {
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        }
                    },
                },
            ],
        },
    ]


# =============================================================================
# MACS Task Fixtures
# =============================================================================


@pytest.fixture
def sample_task():
    """Sample MACS task with typical structure."""
    return Task(
        query="Book a flight to NYC",
        environment_data={
            "tools": [
                {
                    "tool_name": "flight_tools",
                    "actions": [
                        {"name": "search_flights", "description": "Search flights"},
                    ],
                }
            ]
        },
        evaluation_data={
            "assertions": [
                "user: Booking confirmed",
                "agent: Database updated",
            ]
        },
        metadata={"scenario": "Business trip to NYC"},
    )


@pytest.fixture
def sample_task_no_scenario():
    """Task without scenario in metadata."""
    return Task(
        query="Test query",
        environment_data={"tools": []},
        evaluation_data={"assertions": []},
        metadata={},
    )


@pytest.fixture
def sample_task_no_assertions():
    """Task with no assertions."""
    return Task(
        query="Simple query",
        environment_data={},
        evaluation_data={"assertions": []},
        metadata={"scenario": "Simple scenario"},
    )


@pytest.fixture
def travel_task():
    """Detailed travel domain task for integration tests."""
    return Task(
        query="I need to book a flight from San Francisco to New York for next Monday.",
        environment_data={
            "tools": [
                {
                    "tool_name": "travel_tools",
                    "actions": [
                        {
                            "name": "search_flights",
                            "description": "Search for available flights between cities",
                            "input_schema": {
                                "properties": {
                                    "origin": {"type": "string", "description": "Origin city or airport code"},
                                    "destination": {"type": "string", "description": "Destination city or airport code"},
                                    "date": {"type": "string", "description": "Travel date"},
                                }
                            },
                        },
                        {
                            "name": "book_flight",
                            "description": "Book a specific flight",
                            "input_schema": {
                                "properties": {
                                    "flight_id": {"type": "string", "description": "Flight identifier"},
                                    "passenger_name": {"type": "string", "description": "Passenger name"},
                                }
                            },
                        },
                    ],
                }
            ]
        },
        evaluation_data={
            "assertions": [
                "user: The user's flight booking request was acknowledged",
                "user: The user received flight options or a confirmation",
                "agent: The search_flights tool was called with correct parameters",
            ]
        },
        metadata={
            "scenario": """Goal: The user wants to book a flight from San Francisco to New York.

Background:
* User's name is Alice Johnson
* User is a business traveler
* User prefers morning flights
* User has Delta SkyMiles membership""",
            "category": "travel",
            "complexity": "simple",
        },
    )


@pytest.fixture
def macs_task_collection(sample_task, travel_task):
    """Collection of MACS tasks for benchmark.run() tests."""
    return TaskCollection.from_list([sample_task, travel_task])


# =============================================================================
# MACS Task Data Fixtures (for Environment creation)
# =============================================================================


@pytest.fixture
def sample_task_data(sample_tool_specs):
    """Sample task data dict for Environment creation."""
    return {
        "environment_data": {
            "tools": sample_tool_specs,
        }
    }


# =============================================================================
# MACS Agent Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_agent_data():
    """Sample MACS agent configuration."""
    return {
        "agents": [
            {
                "agent_id": "supervisor",
                "agent_name": "Supervisor",
                "agent_instruction": "Coordinate agents",
                "tools": ["flight_tools"],
            }
        ],
        "primary_agent_id": "supervisor",
    }


@pytest.fixture
def sample_agent_spec_flight():
    """Agent spec with only flight tools."""
    return {
        "agent_id": "flight_agent",
        "agent_name": "Flight Agent",
        "tools": ["flight_tools"],
    }


@pytest.fixture
def sample_agent_spec_all():
    """Agent spec with all tools."""
    return {
        "agent_id": "supervisor",
        "agent_name": "Supervisor Agent",
        "tools": ["flight_tools", "hotel_tools"],
    }


@pytest.fixture
def sample_agent_spec_none():
    """Agent spec with no matching tools."""
    return {
        "agent_id": "router",
        "agent_name": "Router Agent",
        "tools": ["unknown_tools"],
    }


# =============================================================================
# MACS Trace and History Fixtures
# =============================================================================


@pytest.fixture
def sample_trace():
    """Sample conversation trace."""
    return MessageHistory(
        [
            {"role": "user", "content": "I need to book a flight to New York"},
            {"role": "assistant", "content": "Sure! When would you like to travel?"},
            {"role": "user", "content": "Next Monday"},
            {"role": "assistant", "content": "I found a flight. Your confirmation number is ABC123."},
        ]
    )


@pytest.fixture
def sample_tool_traces():
    """Sample tool invocation traces."""
    return {
        "search_flights": {
            "invocations": [
                {
                    "inputs": {"origin": "LAX", "destination": "JFK"},
                    "outputs": "Found 3 flights",
                    "status": "success",
                }
            ]
        },
        "book_flight": {
            "invocations": [
                {
                    "inputs": {"flight_id": "AA123"},
                    "outputs": "Booking confirmed",
                    "status": "success",
                }
            ]
        },
    }


@pytest.fixture
def sample_conversation():
    """Sample multi-turn conversation."""
    return MessageHistory(
        [
            {"role": "user", "content": "I need to book a flight from San Francisco to New York for next Monday."},
            {"role": "assistant", "content": "I'll search for flights for you. What time do you prefer to depart?"},
            {"role": "user", "content": "Morning, preferably around 8am."},
            {
                "role": "assistant",
                "content": "I found a Delta flight departing at 8:15am. The fare is $450. Would you like to book this?",
            },
            {"role": "user", "content": "Yes, please book it."},
            {
                "role": "assistant",
                "content": "Your flight has been booked. Confirmation number: DL123456.",
            },
        ]
    )


# =============================================================================
# MACS Scenario Fixtures
# =============================================================================


@pytest.fixture
def sample_scenario():
    """Sample MACS scenario with background."""
    return """Goal: The user wants to book a flight to New York for a business meeting.

Background:
* User's name is John Smith
* User is a frequent business traveler
* User has preferred airline status with Delta
* User prefers aisle seats"""


@pytest.fixture
def minimal_scenario():
    """Minimal scenario without background section."""
    return "User wants to order food for delivery."


@pytest.fixture
def initial_prompt():
    """Sample initial user query."""
    return "I need to book a flight to New York for Monday."


# =============================================================================
# MACS Benchmark Fixtures
# =============================================================================


@pytest.fixture
def macs_benchmark(sample_agent_data, dummy_model):
    """Create a MACS benchmark with dummy model for testing.

    Uses dummy_model from parent conftest.py.
    """
    return ConcreteMACSBenchmark(sample_agent_data, dummy_model)
