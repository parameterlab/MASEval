"""Shared fixtures for MASEval tests."""

import pytest
from typing import Any, Dict, List, Optional, Sequence, Tuple
from maseval import (
    Benchmark,
    AgentAdapter,
    Environment,
    User,
    Task,
    TaskCollection,
    Evaluator,
    MessageHistory,
)
from maseval.core.model import ModelAdapter


# ==================== Dummy Components ====================


class DummyModelAdapter(ModelAdapter):
    """Minimal model adapter for testing."""

    def __init__(self, model_id: str = "test-model", responses: Optional[List[str]] = None):
        super().__init__()
        self._model_id = model_id
        self._responses = responses or ["test response"]
        self._call_count = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def _generate_impl(self, prompt: str, generation_params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return response


class DummyAgent:
    """Minimal agent for testing."""

    def __init__(self, name: str = "dummy_agent"):
        self.name = name
        self.calls = []

    def run(self, query: str) -> str:
        self.calls.append(query)
        return f"Response to: {query}"


class FakeSmolagentsModel:
    """Fake model compatible with smolagents' expectations.

    This model exposes the interface that smolagents' CodeAgent expects:
    - .model_id property
    - .generate() method that returns an object with .content and .token_usage

    This is used for testing framework adapters without requiring real LLM calls.
    """

    def __init__(self, responses: Optional[List[str]] = None):
        self.model_id = "mock-model"
        self._responses = responses or ["mock response"]
        self._call_index = 0

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generate a response compatible with smolagents."""
        text = self._responses[self._call_index % len(self._responses)]
        self._call_index += 1

        class TokenUsage:
            def __init__(self):
                self.input_tokens = 0
                self.output_tokens = 0
                self.total_tokens = 0

        class GenerationResult:
            def __init__(self, text):
                # smolagents expects a ChatMessage-like object with `.content`
                # and a `.token_usage` attribute.
                self.content = text
                self.token_usage = TokenUsage()

            def __str__(self):
                return self.content

        return GenerationResult(text)


class DummyAgentAdapter(AgentAdapter):
    """Test agent adapter that populates message history."""

    def _run_agent(self, query: str) -> str:
        import time

        # Create message history
        history = MessageHistory()
        history.add_message(role="user", content=query)

        # Track timing
        start_time = time.time()

        # Run underlying agent
        response = self.agent.run(query)
        history.add_message(role="assistant", content=response)

        # Store history
        self.set_message_history(history)

        # Populate logs to fulfill contract
        duration = time.time() - start_time
        self.logs.append(
            {
                "query": query,
                "duration_seconds": duration,
                "status": "success",
                "response": response,
            }
        )

        # Return final answer (not the history)
        return response


class DummyEnvironment(Environment):
    """Minimal environment for testing."""

    def setup_state(self, task_data: dict) -> Any:
        return task_data.copy()

    def create_tools(self) -> list:
        return []


class DummyUser(User):
    """Minimal user simulator for testing."""

    def __init__(self, name: str, model: ModelAdapter, **kwargs):
        # Initialize with minimal requirements
        self.name = name
        self.model = model
        self.user_profile = kwargs.get("user_profile", {})
        self.scenario = kwargs.get("scenario", "test scenario")
        self.history = MessageHistory([{"role": "user", "content": kwargs.get("initial_prompt", "Hello")}])
        # Don't initialize simulator to avoid LLM calls in tests

    def get_tool(self) -> Any:
        """Return a dummy tool for testing."""
        return None

    def gather_traces(self) -> dict:
        """Return minimal traces for testing."""
        from datetime import datetime

        return {
            "type": self.__class__.__name__,
            "gathered_at": datetime.now().isoformat(),
            "name": self.name,
            "message_count": len(self.history),
            "history": self.history.to_list(),
        }

    def gather_config(self) -> dict:
        """Return minimal config for testing."""
        from datetime import datetime

        return {
            "type": self.__class__.__name__,
            "gathered_at": datetime.now().isoformat(),
            "name": self.name,
            "user_profile": self.user_profile,
            "scenario": self.scenario,
        }


class DummyEvaluator(Evaluator):
    """Minimal evaluator for testing."""

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.task = task
        self.environment = environment
        self.user = user

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        return {"score": 1.0, "passed": True}


class DummyBenchmark(Benchmark):
    """Minimal benchmark for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_environment_calls = []
        self.setup_user_calls = []
        self.setup_agents_calls = []
        self.setup_evaluators_calls = []
        self.run_agents_calls = []
        self.evaluate_calls = []

    def setup_environment(self, agent_data: Dict[str, Any], task: Task) -> Environment:
        self.setup_environment_calls.append((agent_data, task))
        return DummyEnvironment(task.environment_data)

    def setup_user(self, agent_data: Dict[str, Any], environment: Environment, task: Task) -> Optional[User]:
        self.setup_user_calls.append((agent_data, environment, task))
        return None

    def setup_agents(
        self, agent_data: Dict[str, Any], environment: Environment, task: Task, user: Optional[User]
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        self.setup_agents_calls.append((agent_data, environment, task, user))
        agent = DummyAgent()
        wrapper = DummyAgentAdapter(agent, "test_agent")
        return [wrapper], {"test_agent": wrapper}

    def setup_evaluators(
        self, environment: Environment, task: Task, agents: Sequence[AgentAdapter], user: Optional[User]
    ) -> Sequence[Evaluator]:
        self.setup_evaluators_calls.append((environment, task, agents, user))
        return [DummyEvaluator(task, environment, user)]

    def run_agents(self, agents: Sequence[AgentAdapter], task: Task, environment: Environment) -> Any:
        self.run_agents_calls.append((agents, task, environment))
        # Run the first agent and return final answer
        return agents[0].run(task.query)

    def evaluate(
        self, evaluators: Sequence[Evaluator], agents: Dict[str, AgentAdapter], final_answer: Any, traces: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        self.evaluate_calls.append((evaluators, agents, final_answer, traces))
        # Get first agent's messages (works with any agent name)
        first_agent = next(iter(agents.values())) if agents else None
        if first_agent:
            return [evaluator(first_agent.get_messages()) for evaluator in evaluators]
        return [{"passed": True, "score": 1.0} for _ in evaluators]


# ==================== Fixtures ====================


@pytest.fixture
def dummy_model():
    """Create a dummy model adapter."""
    return DummyModelAdapter()


@pytest.fixture
def dummy_agent():
    """Create a dummy agent."""
    return DummyAgent()


@pytest.fixture
def dummy_agent_adapter(dummy_agent):
    """Create a dummy agent adapter."""
    return DummyAgentAdapter(dummy_agent, "test_agent")


@pytest.fixture
def dummy_environment():
    """Create a dummy environment."""
    return DummyEnvironment({"test_key": "test_value"})


@pytest.fixture
def dummy_user(dummy_model):
    """Create a dummy user."""
    return DummyUser(
        name="test_user",
        model=dummy_model,
        user_profile={"role": "tester"},
        scenario="test scenario",
        initial_prompt="Hello",
    )


@pytest.fixture
def dummy_task():
    """Create a single dummy task."""
    return Task(
        query="Test query",
        environment_data={"key": "value"},
        evaluation_data={"expected": "result"},
        metadata={"difficulty": "easy"},
    )


@pytest.fixture
def dummy_task_collection():
    """Create a collection of dummy tasks."""
    return TaskCollection.from_list(
        [
            {"query": "Query 1", "environment_data": {"task": 1}},
            {"query": "Query 2", "environment_data": {"task": 2}},
            {"query": "Query 3", "environment_data": {"task": 3}},
        ]
    )


@pytest.fixture
def simple_benchmark(dummy_task_collection):
    """Create a simple benchmark instance with tasks.

    Returns:
        tuple: (benchmark, tasks) - Call as benchmark.run(tasks)
    """
    benchmark = DummyBenchmark(agent_data={"model": "test"})
    return benchmark, dummy_task_collection


@pytest.fixture
def agent_data():
    """Create sample agent configuration data."""
    return {"model": "test-model", "temperature": 0.7, "max_tokens": 100}
