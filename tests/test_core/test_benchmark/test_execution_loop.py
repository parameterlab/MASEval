"""Tests for Benchmark.execution_loop() method.

These tests verify the agent-user interaction orchestration, including:
- Query source priority (user initial_prompt vs get_initial_query vs task.query)
- Multi-turn interaction with max_invocations
- Early stopping when user.is_done() returns True
- Message recording (final_answer attached to user traces)
"""

import pytest
from typing import Any, List, Optional, Tuple
import warnings

from maseval import Benchmark, Task, TaskCollection, User


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class ExecutionLoopBenchmark(Benchmark):
    """Benchmark implementation for testing execution_loop behavior."""

    def __init__(self, *args, return_user: Optional[User] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._return_user = return_user
        self.run_agents_calls: List[Tuple[Any, ...]] = []

    def setup_environment(self, agent_data, task):
        from conftest import DummyEnvironment

        return DummyEnvironment(task.environment_data)

    def setup_user(self, agent_data, environment, task):
        return self._return_user

    def setup_agents(self, agent_data, environment, task, user):
        from conftest import DummyAgent, DummyAgentAdapter

        agent = DummyAgent()
        adapter = DummyAgentAdapter(agent, "test_agent")
        return [adapter], {"test_agent": adapter}

    def setup_evaluators(self, environment, task, agents, user):
        from conftest import DummyEvaluator

        return [DummyEvaluator(task, environment, user)]

    def run_agents(self, agents, task, environment, query):
        self.run_agents_calls.append((agents, task, environment, query))
        return agents[0].run(query)

    def evaluate(self, evaluators, agents, final_answer, traces):
        return [{"score": 1.0}]


# =============================================================================
# Tests: Execution Loop Without User
# =============================================================================


@pytest.mark.core
class TestExecutionLoopNoUser:
    """Tests for execution_loop without user simulator."""

    def test_uses_task_query_without_user(self, dummy_model):
        """Uses task.query when no user present."""
        task = Task(query="What is the weather?", environment_data={})
        benchmark = ExecutionLoopBenchmark(agent_data={}, return_user=None)

        env = benchmark.setup_environment({}, task)
        agents, _ = benchmark.setup_agents({}, env, task, None)

        benchmark.execution_loop(agents, task, env, user=None)

        assert len(benchmark.run_agents_calls) == 1
        _, _, _, query = benchmark.run_agents_calls[0]
        assert query == "What is the weather?"

    def test_single_invocation_without_user(self, dummy_model):
        """Single agent run without user (default max_invocations=1)."""
        task = Task(query="Query", environment_data={})
        benchmark = ExecutionLoopBenchmark(agent_data={}, return_user=None)

        env = benchmark.setup_environment({}, task)
        agents, _ = benchmark.setup_agents({}, env, task, None)

        benchmark.execution_loop(agents, task, env, user=None)

        assert len(benchmark.run_agents_calls) == 1

    def test_returns_final_answer(self, dummy_model):
        """Returns final answer from agent."""
        task = Task(query="Test query", environment_data={})
        benchmark = ExecutionLoopBenchmark(agent_data={}, return_user=None)

        env = benchmark.setup_environment({}, task)
        agents, _ = benchmark.setup_agents({}, env, task, None)

        result = benchmark.execution_loop(agents, task, env, user=None)

        assert result is not None
        assert "Response to:" in result


# =============================================================================
# Tests: Execution Loop With User
# =============================================================================


@pytest.mark.core
class TestExecutionLoopWithUser:
    """Tests for execution_loop with user simulator."""

    def test_uses_user_initial_prompt(self, dummy_model):
        """Uses user's initial_prompt as first query."""
        from conftest import DummyUser

        task = Task(query="Task query (should not be used)", environment_data={})
        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="User's initial message",
            max_turns=5,
        )
        benchmark = ExecutionLoopBenchmark(agent_data={}, return_user=user)

        env = benchmark.setup_environment({}, task)
        agents, _ = benchmark.setup_agents({}, env, task, user)

        benchmark.execution_loop(agents, task, env, user=user)

        # First query should be from user's initial_prompt, not task.query
        _, _, _, query = benchmark.run_agents_calls[0]
        assert query == "User's initial message"

    def test_uses_get_initial_query_if_no_prompt(self, dummy_model):
        """Calls get_initial_query() if no initial_prompt."""
        from conftest import DummyUser

        task = Task(query="Task query", environment_data={})
        user = DummyUser(name="test", model=dummy_model, max_turns=5)
        # No initial_prompt, so messages is empty
        user.simulator.return_value = "LLM generated initial query"

        benchmark = ExecutionLoopBenchmark(agent_data={}, return_user=user)

        env = benchmark.setup_environment({}, task)
        agents, _ = benchmark.setup_agents({}, env, task, user)

        benchmark.execution_loop(agents, task, env, user=user)

        # First query should be LLM-generated
        _, _, _, query = benchmark.run_agents_calls[0]
        assert query == "LLM generated initial query"

    def test_multi_turn_interaction(self, dummy_model):
        """Multiple agent-user exchanges up to max_invocations."""
        from conftest import DummyUser

        task = Task(query="Task query", environment_data={})
        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="Start",
            max_turns=5,
        )
        # User responds with different messages each turn
        user.simulator.side_effect = ["Turn 1 response", "Turn 2 response", "Turn 3 response"]

        benchmark = ExecutionLoopBenchmark(
            agent_data={},
            return_user=user,
            max_invocations=3,
        )

        env = benchmark.setup_environment({}, task)
        agents, _ = benchmark.setup_agents({}, env, task, user)

        benchmark.execution_loop(agents, task, env, user=user)

        # Should have 3 agent invocations
        assert len(benchmark.run_agents_calls) == 3

        # Queries should be: initial, then user responses
        queries = [call[3] for call in benchmark.run_agents_calls]
        assert queries[0] == "Start"  # Initial prompt
        assert queries[1] == "Turn 1 response"
        assert queries[2] == "Turn 2 response"

    def test_stops_when_user_done_via_max_turns(self, dummy_model):
        """Stops early when user.is_done() returns True (max_turns reached first)."""
        from conftest import DummyUser

        task = Task(query="Task query", environment_data={})
        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="Start",
            max_turns=2,  # User done after 2 turns (limiting factor)
        )
        user.simulator.side_effect = ["Response 1", "Response 2", "Response 3"]

        benchmark = ExecutionLoopBenchmark(
            agent_data={},
            return_user=user,
            max_invocations=5,  # Would allow 5, but user stops at 2
        )

        env = benchmark.setup_environment({}, task)
        agents, _ = benchmark.setup_agents({}, env, task, user)

        benchmark.execution_loop(agents, task, env, user=user)

        # max_turns=2 is the limiting factor, so exactly 2 invocations
        # Iteration 1: agent runs, simulate_response → turn_count=1, is_done? No
        # Iteration 2: agent runs, simulate_response → turn_count=2, is_done? Yes → break
        assert len(benchmark.run_agents_calls) == 2

    def test_stops_when_user_done_via_stop_token(self, dummy_model):
        """Stops early when user.is_done() returns True (stop_token)."""
        from conftest import DummyUser

        task = Task(query="Task query", environment_data={})
        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="Start",
            max_turns=10,
            stop_token="</stop>",
        )
        # User stops on second response
        user.simulator.side_effect = ["Continue please", "Thanks! </stop>"]

        benchmark = ExecutionLoopBenchmark(
            agent_data={},
            return_user=user,
            max_invocations=5,
        )

        env = benchmark.setup_environment({}, task)
        agents, _ = benchmark.setup_agents({}, env, task, user)

        benchmark.execution_loop(agents, task, env, user=user)

        # Should stop after user says </stop>
        assert len(benchmark.run_agents_calls) == 2

    def test_final_answer_in_user_messages(self, dummy_model):
        """Agent's final_answer is recorded in user messages."""
        from conftest import DummyUser

        task = Task(query="Task query", environment_data={})
        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="Help me",
            max_turns=1,
        )
        user.simulator.return_value = "Thanks"

        benchmark = ExecutionLoopBenchmark(agent_data={}, return_user=user)

        env = benchmark.setup_environment({}, task)
        agents, _ = benchmark.setup_agents({}, env, task, user)

        benchmark.execution_loop(agents, task, env, user=user)

        # User messages should include the agent's response
        messages = list(user.messages)
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) >= 1
        # The assistant message should be the agent's response
        assert "Response to:" in assistant_messages[0]["content"]

    def test_user_response_becomes_next_query(self, dummy_model):
        """User's response is passed to next agent invocation."""
        from conftest import DummyUser

        task = Task(query="Task query", environment_data={})
        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="Initial",
            max_turns=3,  # Allow 3 turns
        )
        # Need 3 responses: after invocation 1, 2, and 3
        user.simulator.side_effect = ["User reply 1", "User reply 2", "User reply 3"]

        benchmark = ExecutionLoopBenchmark(
            agent_data={},
            return_user=user,
            max_invocations=3,  # Will stop after 3 due to max_invocations
        )

        env = benchmark.setup_environment({}, task)
        agents, _ = benchmark.setup_agents({}, env, task, user)

        benchmark.execution_loop(agents, task, env, user=user)

        # Should have 3 invocations limited by max_invocations
        assert len(benchmark.run_agents_calls) == 3

        # First query is initial prompt
        _, _, _, query1 = benchmark.run_agents_calls[0]
        assert query1 == "Initial"

        # Second invocation should use first user reply
        _, _, _, query2 = benchmark.run_agents_calls[1]
        assert query2 == "User reply 1"

        # Third invocation should use second user reply
        _, _, _, query3 = benchmark.run_agents_calls[2]
        assert query3 == "User reply 2"


# =============================================================================
# Tests: Max Invocations Configuration
# =============================================================================


@pytest.mark.core
class TestMaxInvocations:
    """Tests for max_invocations parameter."""

    def test_default_max_invocations_is_one(self):
        """Default is single invocation."""
        benchmark = ExecutionLoopBenchmark(agent_data={})
        assert benchmark.max_invocations == 1

    def test_custom_max_invocations(self):
        """Custom max_invocations is stored."""
        benchmark = ExecutionLoopBenchmark(agent_data={}, max_invocations=5)
        assert benchmark.max_invocations == 5

    def test_warning_max_invocations_without_user(self):
        """Warning issued when max_invocations > 1 but no user."""
        task = Task(query="Test", environment_data={})
        benchmark = ExecutionLoopBenchmark(
            agent_data={},
            return_user=None,
            max_invocations=5,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            benchmark.run(TaskCollection([task]))

            # Check for warning about max_invocations without user
            warning_messages = [str(warning.message) for warning in w]
            assert any("max_invocations" in msg and "no user simulator" in msg.lower() for msg in warning_messages)


# =============================================================================
# Tests: Integration with run()
# =============================================================================


@pytest.mark.core
class TestBenchmarkRunWithUser:
    """Tests for run() integration with users."""

    def test_run_with_user_uses_execution_loop(self, dummy_model):
        """run() delegates to execution_loop."""
        from conftest import DummyUser

        task = Task(query="Task query", environment_data={})
        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="User query",
            max_turns=1,
        )
        user.simulator.return_value = "Done"

        benchmark = ExecutionLoopBenchmark(agent_data={}, return_user=user)

        benchmark.run(TaskCollection([task]))

        # Verify run_agents was called with user's initial prompt
        assert len(benchmark.run_agents_calls) == 1
        _, _, _, query = benchmark.run_agents_calls[0]
        assert query == "User query"

    def test_complete_traces_with_user(self, dummy_model):
        """Traces include complete user conversation."""
        from conftest import DummyUser

        task = Task(query="Task query", environment_data={})
        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_prompt="Hello",
            max_turns=2,
        )
        user.simulator.side_effect = ["Reply 1", "Reply 2"]

        benchmark = ExecutionLoopBenchmark(
            agent_data={},
            return_user=user,
            max_invocations=2,
        )

        reports = benchmark.run(TaskCollection([task]))

        # Check that user traces are in the report
        assert len(reports) == 1
        traces = reports[0]["traces"]
        assert "user" in traces

        # User traces should have the conversation
        user_traces = traces["user"]
        assert "messages" in user_traces
        # Should have exactly: initial + 2 exchanges (initial, agent1, user1, agent2, user2)
        assert user_traces["message_count"] == 5

        # Verify exact message sequence
        messages = user_traces["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert "Response to:" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Reply 1"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"
        assert messages[4]["content"] == "Reply 2"
