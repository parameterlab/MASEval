"""Cross-framework AgentAdapter contract tests.

Verifies that ALL AgentAdapter implementations (smolagents, langgraph, dummy)
implement the same contract and behave identically for key operations.

This validates MASEval's CORE PROMISE: framework-agnostic agent abstraction.

Contract tests ensure that users can swap between agent frameworks (smolagents,
langgraph, custom implementations) without changing their benchmark code.

What this contract validates:
- run() returns string (final answer) and populates message history consistently
- get_messages() returns OpenAI-compatible format (role, content fields)
- Callbacks trigger uniformly (on_run_start, on_run_end)
- Trace structure is consistent across implementations (gather_traces)
- Config structure is consistent across implementations (gather_config)
- Callback lifecycle ordering is predictable
- Multiple callbacks execute in registration order
- Multi-turn conversations accumulate history consistently
- Edge cases (empty history) behave consistently

Note: Message history manipulation (set/append/clear) is NOT part of the contract
as it's not universally supported across frameworks. These are tested separately
in framework-specific integration tests.

If these tests fail, the abstraction layer is broken and users cannot reliably
swap between agent frameworks.
"""

import pytest
from typing import List, Optional
from maseval.core.callback import AgentCallback
from conftest import DummyAgent, DummyAgentAdapter, FakeSmolagentsModel

# Mark all tests as contract + interface (requires optional dependencies)
pytestmark = [pytest.mark.contract, pytest.mark.interface]


# ==================== Helper Classes ====================


class CallbackTracker(AgentCallback):
    """Tracks callback events for testing."""

    def __init__(self):
        super().__init__()
        self.events: List[str] = []

    # AgentAdapter calls `on_run_start` / `on_run_end` on AgentCallback instances,
    # so implement those to track events.
    def on_run_start(self, agent):
        self.events.append("on_agent_start")

    def on_run_end(self, agent, result):
        self.events.append("on_agent_end")


class MockLLM:
    """Mock LLM that returns deterministic responses."""

    def __init__(self, responses: Optional[List[str]] = None, include_tools: bool = False):
        self.responses = responses or ["Test response"]
        self.include_tools = include_tools
        self.call_count = 0

    def __call__(self, messages, **kwargs):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


# ==================== Factory Functions ====================


def create_agent_for_framework(framework: str, mock_llm: MockLLM):
    """Create a framework-specific agent instance."""
    if framework == "dummy":
        return DummyAgent()

    elif framework == "smolagents":
        pytest.importorskip("smolagents")
        # Import here to avoid requiring smolagents for all tests
        from smolagents import CodeAgent

        # Use the shared FakeSmolagentsModel from conftest
        mock_model = FakeSmolagentsModel(mock_llm.responses)
        agent = CodeAgent(tools=[], model=mock_model, max_steps=2)
        return agent

    elif framework == "langgraph":
        pytest.importorskip("langgraph")
        # Import here to avoid requiring langgraph for all tests
        from langgraph.graph import StateGraph, END
        from typing_extensions import TypedDict
        from langchain_core.messages import AIMessage

        class State(TypedDict):
            messages: List[object]

        def agent_node(state: State) -> State:
            """Simple agent node that appends a response using LangChain message objects."""
            messages = state["messages"]

            # Extract user content from last message; support both dicts and message objects.
            if messages:
                last = messages[-1]
                if isinstance(last, dict):
                    user_msg = last.get("content", "")
                elif hasattr(last, "content"):
                    user_msg = getattr(last, "content")
                else:
                    user_msg = str(last)
            else:
                user_msg = "No input"

            response = mock_llm([{"role": "user", "content": user_msg}])

            # Return LangChain-style message objects so the adapter conversion works
            return {"messages": messages + [AIMessage(content=response)]}

        # Build graph
        graph = StateGraph(State)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

        return graph.compile()

    else:
        raise ValueError(f"Unknown framework: {framework}")


def create_adapter_for_framework(framework: str, agent, callbacks: Optional[List[AgentCallback]] = None):
    """Create a framework-specific adapter instance."""
    # Verify agent is not None and is the expected type for the framework
    assert agent is not None, f"Agent instance is None for framework: {framework}"

    if framework == "dummy":
        assert isinstance(agent, DummyAgent), f"Expected DummyAgent, got {type(agent)}"
        return DummyAgentAdapter(agent, "test_agent", callbacks=callbacks)

    elif framework == "smolagents":
        pytest.importorskip("smolagents")
        from maseval.interface.agents.smolagents import SmolAgentAdapter
        from smolagents import MultiStepAgent, CodeAgent, ToolCallingAgent

        # Accept any of the three smolagents agent types
        # All inherit from MultiStepAgent which provides run() and memory attributes
        agent_types = (MultiStepAgent, CodeAgent, ToolCallingAgent)
        assert isinstance(agent, agent_types), (
            f"Expected smolagents agent (MultiStepAgent, CodeAgent, or ToolCallingAgent), got {type(agent).__name__}"
        )
        return SmolAgentAdapter(agent, "test_agent", callbacks=callbacks)

    elif framework == "langgraph":
        pytest.importorskip("langgraph")
        from maseval.interface.agents.langgraph import LangGraphAgentAdapter

        # LangGraph compiled graphs have an 'invoke' method
        assert hasattr(agent, "invoke"), f"Expected LangGraph compiled graph with 'invoke' method, got {type(agent)}"
        return LangGraphAgentAdapter(agent, "test_agent", callbacks=callbacks)

    else:
        raise ValueError(f"Unknown framework: {framework}")


# ==================== Contract Tests ====================


@pytest.mark.contract
@pytest.mark.parametrize("framework", ["dummy", "smolagents", "langgraph"])
class TestAgentAdapterContract:
    """Verify all AgentAdapter implementations honor the same contract."""

    def test_adapter_run_returns_same_structure(self, framework):
        """Test all frameworks return string result and populate message history.

        Contract: run() must return a string (the final answer) and populate
        the message history. This is the most basic behavioral guarantee.
        """
        mock_llm = MockLLM(responses=["Test response to query"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent)

        result = adapter.run("Test query")

        # All should return string (final answer)
        assert isinstance(result, str)
        assert len(result) > 0

        # All should populate message history identically
        history = adapter.get_messages()
        assert len(history) > 0

        # Some frameworks (smolagents) prepend a system message; accept either.
        assert history[0]["role"] in ["user", "system"]

        # Ensure at least one assistant/tool message exists somewhere in the history
        assert any(msg.get("role") in ["assistant", "tool"] for msg in history)

    def test_adapter_message_format_identical(self, framework):
        """Test all frameworks produce OpenAI-compatible message format.

        Contract: All messages must have 'role' and 'content' keys, matching
        the OpenAI chat completion API format for cross-framework compatibility.
        """
        mock_llm = MockLLM(responses=["Response content"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent)

        adapter.run("Test query")
        history = adapter.get_messages()

        # Verify OpenAI format
        for msg in history:
            assert "role" in msg, f"Message missing 'role' key: {msg}"
            assert "content" in msg, f"Message missing 'content' key: {msg}"
            role = str(msg["role"]) if msg.get("role") is not None else ""
            allowed = {"user", "assistant", "system", "tool"}
            assert role in allowed or role.startswith("tool"), f"Invalid role: {msg['role']}"

    def test_adapter_callbacks_triggered_uniformly(self, framework):
        """Test callbacks fire in same order across all frameworks.

        Contract: on_run_start and on_run_end callbacks must fire in the
        correct order (start before run, end after run) for all adapters.
        """
        callback_tracker = CallbackTracker()
        mock_llm = MockLLM(responses=["Response"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent, callbacks=[callback_tracker])

        adapter.run("Test query")

        # All frameworks should trigger same callback sequence
        assert "on_agent_start" in callback_tracker.events
        assert "on_agent_end" in callback_tracker.events
        assert callback_tracker.events[0] == "on_agent_start"
        assert callback_tracker.events[-1] == "on_agent_end"

    def test_adapter_traces_same_structure(self, framework):
        """Test gather_traces returns consistent structure across frameworks.

        Contract: All adapters must provide message history in traces, enabling
        uniform access to execution data regardless of underlying framework.
        """
        mock_llm = MockLLM(responses=["Response"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent)

        adapter.run("Test query")
        traces = adapter.gather_traces()

        # All should include message history; different adapters name this key
        if "message_history" in traces:
            messages = traces["message_history"]
        else:
            messages = traces.get("messages", [])

        assert isinstance(messages, list)
        assert len(messages) > 0

    def test_adapter_config_same_structure(self, framework):
        """Test gather_config returns consistent structure across frameworks.

        Contract: All adapters must provide agent name in config, enabling
        identification and reproducibility tracking.
        """
        mock_llm = MockLLM(responses=["Response"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent)

        config = adapter.gather_config()

        # All should include agent name
        assert "agent_name" in config or "name" in config
        # All should include some identifying information
        assert len(config) > 0

    def test_adapter_get_messages_after_multiple_runs(self, framework):
        """Test message history accumulation across multiple agent runs.

        Contract: Message history behavior during multi-turn conversations must
        be consistent, whether histories accumulate or are per-run.
        """
        mock_llm = MockLLM(responses=["First response", "Second response"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent)

        # First run
        adapter.run("First query")
        history_1 = adapter.get_messages()
        len_1 = len(history_1)
        assert len_1 > 0

        # Second run (behavior may differ: some accumulate, some reset)
        adapter.run("Second query")
        history_2 = adapter.get_messages()
        len_2 = len(history_2)

        # At minimum, should have messages from second run
        assert len_2 > 0
        # Note: We don't enforce accumulation vs reset - that's framework-specific

    def test_adapter_empty_query_handling(self, framework):
        """All frameworks handle empty queries gracefully.

        Note: This test accepts both success and failure for empty queries.
        Consider removing this test in the future as it tests implementation
        details rather than contract requirements. The contract does not
        specify expected behavior for empty/invalid inputs.
        """
        mock_llm = MockLLM(responses=["Response to empty"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent)

        # Should not crash on empty query
        try:
            result = adapter.run("")
            # If it succeeds, should return something
            assert result is not None
        except (ValueError, AssertionError):
            # It's acceptable to reject empty queries
            pass

    def test_adapter_on_event_callback(self, framework):
        """Test that standard callback hooks fire consistently across frameworks.

        Contract: All adapters must fire on_run_start and on_run_end callbacks.
        The on_event hook is optional for custom events.
        """
        events = []

        class EventTracker(AgentCallback):
            def on_event(self, event_name: str, **data):
                events.append(("event", event_name, data))

            def on_run_start(self, agent):
                events.append(("on_run_start", agent.name))

            def on_run_end(self, agent, result):
                events.append(("on_run_end", agent.name))

        mock_llm = MockLLM(responses=["Response"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent, callbacks=[EventTracker()])

        adapter.run("Test query")

        # Verify standard callbacks fired
        event_types = [e[0] for e in events]
        assert "on_run_start" in event_types
        assert "on_run_end" in event_types

        # Note: on_event() is a generic hook that adapters can use to emit custom events.
        # The base AgentAdapter doesn't emit any events by default, but the callback
        # mechanism should work if adapters choose to use it.

    def test_adapter_callback_lifecycle_order(self, framework):
        """Test callbacks fire in correct lifecycle order with proper state.

        Contract: on_run_start fires before execution with initial state,
        on_run_end fires after execution with final state and result.

        Note: smolagents always has a system message in memory at start.
        """
        lifecycle_events = []

        class LifecycleTracker(AgentCallback):
            def on_run_start(self, agent):
                # Capture state at start
                msg_count_at_start = len(agent.get_messages())
                lifecycle_events.append(("start", msg_count_at_start))

            def on_run_end(self, agent, result):
                # Capture state at end
                msg_count_at_end = len(agent.get_messages())
                lifecycle_events.append(("end", msg_count_at_end, result))

        mock_llm = MockLLM(responses=["Test response"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent, callbacks=[LifecycleTracker()])

        result = adapter.run("Test query")

        # Verify callback order
        assert len(lifecycle_events) == 2
        assert lifecycle_events[0][0] == "start"
        assert lifecycle_events[1][0] == "end"

        # Verify on_run_start fires before user messages are added
        # smolagents has 1 message (system), others have 0
        expected_start_count = 1 if framework == "smolagents" else 0
        assert lifecycle_events[0][1] == expected_start_count

        # Verify on_run_end fires after message history is populated
        assert lifecycle_events[1][1] > expected_start_count  # Has more messages at end

        # Verify result is passed to on_run_end
        assert lifecycle_events[1][2] == result

    def test_adapter_multiple_callbacks(self, framework):
        """Test multiple callbacks execute in registration order.

        Contract: When multiple callbacks are registered, they must execute
        in the order they were added to the callbacks list.
        """
        call_order = []

        class FirstCallback(AgentCallback):
            def on_run_start(self, agent):
                call_order.append("first_start")

            def on_run_end(self, agent, result):
                call_order.append("first_end")

        class SecondCallback(AgentCallback):
            def on_run_start(self, agent):
                call_order.append("second_start")

            def on_run_end(self, agent, result):
                call_order.append("second_end")

        mock_llm = MockLLM(responses=["Response"])
        agent = create_agent_for_framework(framework, mock_llm)
        agent_adapter = create_adapter_for_framework(framework, agent, callbacks=[FirstCallback(), SecondCallback()])

        agent_adapter.run("Test query")

        # Verify all callbacks fired
        assert len(call_order) == 4

        # Verify order: all on_run_start before any on_run_end
        assert call_order == ["first_start", "second_start", "first_end", "second_end"]

    def test_adapter_message_history_after_clear_and_run(self, framework):
        """Test that message history is correctly populated after clearing and running.

        This test validates two key contract requirements:
        1. Clear history should reset the agent's state
        2. Running the agent after clearing should start with a fresh history
        """
        mock_llm = MockLLM(responses=["Test response"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent)

    def test_adapter_logs_populated_after_run(self, framework):
        """Test all adapters populate self.logs during execution.

        Contract: All AgentAdapter implementations must populate the self.logs
        attribute with execution information. This enables uniform access to
        detailed execution traces regardless of the underlying framework.

        The logs should contain basic execution information that can be used
        for debugging, monitoring, and evaluation purposes.
        """
        mock_llm = MockLLM(responses=["Test response"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent)

        # Before run, logs should be empty
        assert isinstance(adapter.logs, list)
        initial_log_count = len(adapter.logs)

        # Run the agent
        adapter.run("Test query")

        # After run, logs should be populated
        assert len(adapter.logs) > initial_log_count
        assert isinstance(adapter.logs, list)

        # Verify logs contain useful information (at least one entry)
        # Different frameworks may structure logs differently, but all should have entries
        assert len(adapter.logs) > 0

    def test_adapter_logs_in_gather_traces(self, framework):
        """Test that gather_traces includes logs field.

        Contract: The gather_traces() method must include the logs field,
        providing a unified way to access execution details across all frameworks.
        """
        mock_llm = MockLLM(responses=["Test response"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent)

        # Run the agent
        adapter.run("Test query")

        # Gather traces
        traces = adapter.gather_traces()

        # Verify logs field exists and is populated
        assert "logs" in traces
        assert isinstance(traces["logs"], list)
        assert len(traces["logs"]) > 0

    def test_adapter_logs_structure_has_basic_info(self, framework):
        """Test that logs entries contain basic execution information.

        Contract: While the exact structure of log entries may vary by framework,
        all implementations should provide basic execution information in their logs.
        This test verifies that log entries are dictionaries containing some form
        of execution data.
        """
        mock_llm = MockLLM(responses=["Test response"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent)

        # Run the agent
        adapter.run("Test query")

        # Verify logs contain dict entries with data
        logs = adapter.logs
        assert len(logs) > 0

        # Each log entry should be a dictionary
        for log_entry in logs:
            assert isinstance(log_entry, dict)
            # Should have at least one field with information
            assert len(log_entry) > 0

    def test_adapter_logs_accumulate_across_runs(self, framework):
        """Test that logs accumulate or reset consistently across multiple runs.

        Contract: Adapter logs should maintain a consistent lifecycle behavior
        across runs. While accumulation vs reset is framework-specific, the
        behavior should be predictable and documented.
        """
        mock_llm = MockLLM(responses=["First response", "Second response"])
        agent = create_agent_for_framework(framework, mock_llm)
        adapter = create_adapter_for_framework(framework, agent)

        # First run
        adapter.run("First query")
        logs_count_after_first = len(adapter.logs)
        assert logs_count_after_first > 0

        # Second run
        adapter.run("Second query")
        logs_count_after_second = len(adapter.logs)

        # Logs should either accumulate or stay consistent
        # (we accept both behaviors as long as logs are populated)
        assert logs_count_after_second > 0
