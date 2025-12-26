"""Tau 2 Benchmark - Main Implementation.

Framework-agnostic implementation of the tau2-bench benchmark for evaluating
LLM-based agents on customer service tasks across multiple domains.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Reference Paper: "Tau-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains"
https://arxiv.org/abs/2406.12045

Usage:
    from maseval.benchmark.tau2 import (
        Tau2Benchmark, Tau2Environment, Tau2Evaluator, Tau2User,
        load_tasks, configure_model_ids,
    )

    # Load data and configure model IDs
    tasks = load_tasks("retail", split="base", limit=5)
    configure_model_ids(
        tasks,
        user_model_id="gpt-4o",
        evaluator_model_id="gpt-4o",  # Optional - only for NL assertions
    )

    # Create your framework-specific benchmark subclass
    class MyTau2Benchmark(Tau2Benchmark):
        def setup_agents(self, agent_data, environment, task, user):
            # Your framework-specific agent creation
            ...

        def get_model_adapter(self, model_id, **kwargs):
            # Create and optionally register model adapters
            adapter = MyModelAdapter(model_id)
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    # Run
    benchmark = MyTau2Benchmark(agent_data={})
    results = benchmark.run(tasks)

Default Agent Implementation:
    For comparison with the original tau2-bench results, use DefaultAgentTau2Benchmark
    which implements their agent logic exactly:

    from maseval.benchmark.tau2 import DefaultAgentTau2Benchmark, load_tasks, configure_model_ids

    tasks = load_tasks("retail", split="base", limit=5)
    configure_model_ids(tasks, user_model_id="gpt-4o")

    benchmark = DefaultAgentTau2Benchmark(
        agent_data={"model_id": "gpt-4o"},
    )
    results = benchmark.run(tasks)
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

from maseval import AgentAdapter, Benchmark, Evaluator, ModelAdapter, Task, User
from maseval.core.user import AgenticUser

from maseval.benchmark.tau2.environment import Tau2Environment
from maseval.benchmark.tau2.evaluator import Tau2Evaluator


# =============================================================================
# User Simulator
# =============================================================================


class Tau2User(AgenticUser):
    """Tau2-specific user simulator with customer service personas.

    Extends the AgenticUser class with tau2-specific behavior:
    - Customer personas from user_scenario
    - Domain-aware responses (airline, retail, telecom)
    - Multi-turn interaction support
    - Tool usage capabilities

    Note: This is a base class. Framework-specific subclasses should override
    get_tool() to return a compatible tool.
    """

    DEFAULT_MAX_TURNS = 10  # Higher than MACS due to more complex tasks
    DEFAULT_STOP_TOKEN = "</stop>"
    DEFAULT_EARLY_STOPPING_CONDITION = "The user's issue has been fully resolved by the agent"

    def __init__(
        self,
        model: ModelAdapter,
        scenario: str,
        initial_query: str,
        name: str = "Customer",
        template: Optional[str] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        stop_token: str = DEFAULT_STOP_TOKEN,
        early_stopping_condition: str = DEFAULT_EARLY_STOPPING_CONDITION,
        tools: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize Tau2 user simulator.

        Args:
            model: ModelAdapter for LLM-based response generation
            scenario: Full scenario text containing user instructions
            initial_query: The initial query to the agent
            name: User name for identification (default: "Customer")
            template: Optional custom prompt template
            max_turns: Maximum conversation turns
            stop_token: Token indicating user satisfaction
            early_stopping_condition: Description of when to emit stop token
            tools: Optional dictionary of tools available to the user
        """
        # Extract user profile from scenario
        user_profile = self._extract_user_profile(scenario)

        super().__init__(
            name=name,
            model=model,
            user_profile=user_profile,
            scenario=scenario,
            initial_query=initial_query,
            template=template,
            max_turns=max_turns,
            stop_token=stop_token,
            early_stopping_condition=early_stopping_condition,
            tools=tools,
        )

    def get_tool(self) -> Any:
        """Return a tool for agent interaction.

        This base implementation raises NotImplementedError.
        Framework-specific subclasses should override this method.

        Raises:
            NotImplementedError: Always, as this must be implemented by subclass.
        """
        raise NotImplementedError("Tau2User.get_tool() must be overridden by framework-specific subclass.")

    @staticmethod
    def _extract_user_profile(scenario: str) -> Dict[str, Any]:
        """Extract user profile from scenario text.

        Args:
            scenario: Full scenario/instructions text

        Returns:
            Dict with user profile fields
        """
        profile: Dict[str, Any] = {}

        # Parse structured format if present
        if "Persona:" in scenario or "persona:" in scenario:
            # Try to extract persona section
            for prefix in ["Persona:", "persona:"]:
                if prefix in scenario:
                    parts = scenario.split(prefix, 1)
                    if len(parts) > 1:
                        persona_section = parts[1].split("\n")[0].strip()
                        profile["persona"] = persona_section

        # Include full scenario as context
        profile["full_scenario"] = scenario

        return profile

    def gather_traces(self) -> Dict[str, Any]:
        """Gather traces with Tau2-specific information."""
        traces = super().gather_traces()
        traces.update(
            {
                "max_turns": self.max_turns,
                "turns_used": self._turn_count,
                "stopped_by_user": self._stopped,
            }
        )
        return traces


# =============================================================================
# Benchmark
# =============================================================================


class Tau2Benchmark(Benchmark):
    """Tau2 Benchmark - Framework-agnostic base class.

    This base class handles:
    - Environment setup with Tau2Environment (real tools)
    - Deterministic evaluation via database state comparison
    - Optional user simulation for multi-turn tasks

    Users must subclass and implement:
    - setup_agents() for their agent framework
    - get_model_adapter() to provide model adapters

    Model IDs for components are read from task data:
    - task.user_data["model_id"] for user simulator
    - task.evaluation_data["model_id"] for NL assertion evaluator (optional)

    Use configure_model_ids() to set these values after loading tasks.

    Example:
        class MyTau2Benchmark(Tau2Benchmark):
            def setup_agents(self, agent_data, environment, task, user):
                # Setup your agents here
                ...

            def get_model_adapter(self, model_id, **kwargs):
                return MyModelAdapter(model_id)

        tasks = load_tasks("retail")
        configure_model_ids(tasks, user_model_id="gpt-4o")

        benchmark = MyTau2Benchmark(agent_data={})
        benchmark.run(tasks)
    """

    def __init__(
        self,
        agent_data: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
        n_task_repeats: int = 1,
        max_invocations: int = 10,
        data_dir: Optional[Path] = None,
        **kwargs: Any,
    ):
        """Initialize benchmark.

        Args:
            agent_data: Agent configuration dict
            callbacks: Benchmark callbacks
            n_task_repeats: Repetitions per task (for Pass@k metrics, use k here)
            max_invocations: Maximum agent-user interaction rounds
            data_dir: Base data directory for domain data
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(agent_data or {}, callbacks, n_task_repeats, max_invocations, **kwargs)
        self._data_dir = data_dir

    def _get_user_model_id(self, task: Task) -> str:
        """Get user simulator model ID from task.user_data.

        Raises:
            ValueError: If model_id not configured in task.user_data
        """
        model_id = task.user_data.get("model_id")
        if model_id is None:
            raise ValueError(
                "User simulator model_id not configured in task.user_data.\n"
                "Use configure_model_ids() after loading tasks:\n\n"
                "    from maseval.benchmark.tau2 import load_tasks, configure_model_ids\n\n"
                "    tasks = load_tasks('retail')\n"
                "    configure_model_ids(\n"
                "        tasks,\n"
                "        user_model_id='gpt-4o',\n"
                "    )"
            )
        return model_id

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
    ) -> Tau2Environment:
        """Create environment for a task.

        Creates a Tau2Environment with real tool implementations
        for the task's domain.

        Args:
            agent_data: Agent configuration
            task: Current task

        Returns:
            Tau2Environment instance
        """
        return Tau2Environment(
            task_data=task.environment_data,
            data_dir=self._data_dir,
        )

    def setup_user(  # type: ignore[override]
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
    ) -> Optional[User]:
        """Create Tau2 user simulator.

        Creates a Tau2User with scenario from the task.
        Model ID is read from task.user_data["model_id"].

        Note: Tau2User.get_tool() raises NotImplementedError.
        Framework-specific subclasses should wrap this user
        or override setup_user() to return a user with get_tool() implemented.

        Args:
            agent_data: Agent configuration
            environment: The task environment
            task: Current task with user scenario

        Returns:
            Tau2User instance
        """
        # Build scenario from user instructions
        user_data = task.user_data
        instructions = user_data.get("instructions", {})

        if isinstance(instructions, str):
            scenario = instructions
        elif isinstance(instructions, dict):
            parts = []
            if instructions.get("reason_for_call"):
                parts.append(f"Reason for call: {instructions['reason_for_call']}")
            if instructions.get("known_info"):
                parts.append(f"Known info: {instructions['known_info']}")
            if instructions.get("task_instructions"):
                parts.append(f"Task: {instructions['task_instructions']}")
            scenario = "\n".join(parts)
        else:
            scenario = ""

        # Add persona if available
        persona = user_data.get("persona")
        if persona:
            scenario = f"Persona: {persona}\n\n{scenario}"

        user_model_id = self._get_user_model_id(task)

        # Get user tools from environment
        user_tools = environment.create_user_tools()

        return Tau2User(
            model=self.get_model_adapter(
                user_model_id,
                register_name="user_simulator",
            ),
            scenario=scenario,
            initial_query=task.query,
            tools=user_tools,
        )

    @abstractmethod
    def setup_agents(  # type: ignore[override]
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create agents for this task. Must be implemented by subclass.

        Args:
            agent_data: Agent configuration
            environment: Tau2Environment with real tools
            task: Current task
            user: Optional user simulator

        Returns:
            Tuple of (ordered agent list, agent dict keyed by ID)
        """
        pass

    def setup_evaluators(  # type: ignore[override]
        self,
        environment: Tau2Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
    ) -> Sequence[Evaluator]:
        """Create evaluator for the task.

        Creates a Tau2Evaluator that performs deterministic
        database state comparison.

        Args:
            environment: Tau2Environment instance
            task: Current task with evaluation criteria
            agents: Agent instances
            user: Optional user simulator

        Returns:
            List with single Tau2Evaluator instance
        """
        return [
            Tau2Evaluator(
                task=task,
                environment=environment,
                data_dir=self._data_dir,
            )
        ]

    def run_agents(  # type: ignore[override]
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Tau2Environment,
        query: str = "",
    ) -> Any:
        """Execute agents and return final answer.

        Args:
            agents: Agent instances to run
            task: Current task
            environment: Tau2Environment
            query: Query/prompt for agents

        Returns:
            Final answer from agents
        """
        answers = [agent.run(query) for agent in agents]
        return answers[0] if len(answers) == 1 else answers

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate using Tau2 evaluators.

        Uses each evaluator's filter_traces() method to extract relevant data,
        then calls the evaluator with the filtered traces.

        Returns tau2 format:
        - reward: Float [0.0, 1.0]
        - passed: Boolean
        - reward_breakdown: Per-evaluator scores
        - env_check, action_check, communicate_check: Detailed results

        Args:
            evaluators: List of evaluators
            agents: Dict of agents
            final_answer: Final answer from agents
            traces: Execution traces

        Returns:
            List of evaluation result dicts
        """
        results = []
        for evaluator in evaluators:
            filtered_traces = evaluator.filter_traces(traces)
            result = evaluator(filtered_traces, final_answer)
            results.append(result)

        return results


# =============================================================================
# Default Agent Implementation
# =============================================================================

# Agent system prompt constants (matching original tau2-bench)
_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

_SYSTEM_PROMPT_TEMPLATE = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


class DefaultTau2Agent:
    """Default agent implementation matching original tau2-bench LLMAgent.

    This agent mirrors the behavior of the original tau2-bench LLMAgent class,
    enabling direct comparison with the original benchmark results.

    The agent uses a simple ReAct-style loop:
    1. Receives user message
    2. Generates response (text or tool call)
    3. If tool call: executes tool and loops back to step 2
    4. If text: returns text as response

    Original implementation: tau2-bench/src/tau2/agent/llm_agent.py

    Attributes:
        tools: Dictionary mapping tool names to callables
        policy: Domain policy text (markdown)
        model: ModelAdapter for LLM calls
        llm_args: Additional arguments for LLM calls
        max_tool_calls: Maximum tool calls per turn (prevents infinite loops)
    """

    def __init__(
        self,
        tools: Dict[str, Callable],
        policy: str,
        model: ModelAdapter,
        llm_args: Optional[Dict[str, Any]] = None,
        max_tool_calls: int = 50,
    ):
        """Initialize the default tau2 agent.

        Args:
            tools: Dictionary mapping tool names to callable implementations
            policy: Domain policy text (markdown format)
            model: ModelAdapter for making LLM calls
            llm_args: Optional additional arguments passed to model.generate()
            max_tool_calls: Maximum number of tool calls per agent turn
        """
        self.tools = tools
        self.policy = policy
        self.model = model
        self.llm_args = llm_args or {}
        self.max_tool_calls = max_tool_calls

        # Build system prompt
        self.system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            agent_instruction=_AGENT_INSTRUCTION,
            domain_policy=self.policy,
        )

        # Message history for the conversation
        self._messages: List[Dict[str, Any]] = []
        self._tool_call_count = 0

    def reset(self) -> None:
        """Reset the agent state for a new conversation."""
        self._messages = []
        self._tool_call_count = 0

    def run(self, query: str) -> str:
        """Process a user query and return the agent's response.

        This method handles the full agent turn:
        1. Adds user message to history
        2. Generates LLM response with tool access
        3. If tool call: executes tools and continues generating
        4. Returns final text response to user

        Args:
            query: The user's message/query

        Returns:
            Agent's text response to the user
        """
        # Add user message to history
        self._messages.append({"role": "user", "content": query})

        # Generate response with potential tool calls
        return self._generate_with_tools()

    def _generate_with_tools(self) -> str:
        """Generate response, handling any tool calls.

        Implements the agent's ReAct loop:
        - Generate LLM response with tools available
        - If response includes tool calls, execute them and continue
        - If response is text only, return it

        Returns:
            Final text response from the agent
        """
        while self._tool_call_count < self.max_tool_calls:
            # Build messages for LLM call
            messages = [{"role": "system", "content": self.system_prompt}] + self._messages

            # Generate response with tool access using chat() method
            response = self.model.chat(
                messages=messages,
                tools=self._get_tool_definitions(),
                **self.llm_args,
            )

            # Parse response from ChatResponse
            content = response.content or ""
            tool_calls = response.tool_calls or []

            if tool_calls:
                # Add assistant message with tool calls
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )

                # Execute each tool call
                for tool_call in tool_calls:
                    self._tool_call_count += 1
                    tool_result = self._execute_tool_call(tool_call)

                    # Add tool result to history
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": str(tool_result),
                        }
                    )

                # Continue loop to generate next response
                continue
            else:
                # Text response - add to history and return
                self._messages.append({"role": "assistant", "content": content})
                return content

        # Max tool calls reached - return empty or error message
        return "I apologize, but I've encountered an issue processing your request. Please try again."

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a single tool call.

        Args:
            tool_call: Dict with 'name' and 'arguments' keys

        Returns:
            Tool execution result
        """
        name = tool_call.get("name", "")
        # Handle both 'arguments' (dict) and 'function' (nested dict) formats
        if "function" in tool_call:
            arguments = tool_call["function"].get("arguments", {})
        else:
            arguments = tool_call.get("arguments", {})

        # Handle string arguments (JSON encoded)
        if isinstance(arguments, str):
            import json

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        if name not in self.tools:
            return f"Error: Tool '{name}' not found"

        try:
            result = self.tools[name](**arguments)
            return result
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Generate tool definitions for the LLM.

        Returns:
            List of tool definitions in OpenAI function calling format
        """
        import inspect

        definitions = []
        for name, func in self.tools.items():
            sig = inspect.signature(func)
            doc = func.__doc__ or f"Tool: {name}"

            # Build parameters schema
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                # Determine parameter type and build property schema
                param_schema: Dict[str, Any] = {"description": f"Parameter: {param_name}"}

                if param.annotation is not inspect.Parameter.empty:
                    if param.annotation is int:
                        param_schema["type"] = "integer"
                    elif param.annotation is float:
                        param_schema["type"] = "number"
                    elif param.annotation is bool:
                        param_schema["type"] = "boolean"
                    elif param.annotation is list or (hasattr(param.annotation, "__origin__") and param.annotation.__origin__ is list):
                        param_schema["type"] = "array"
                        # Add items schema for array types (required by Google GenAI)
                        param_schema["items"] = {"type": "string"}
                        # Try to get the inner type for List[X]
                        if hasattr(param.annotation, "__args__") and param.annotation.__args__:
                            inner_type = param.annotation.__args__[0]
                            if inner_type is int:
                                param_schema["items"] = {"type": "integer"}
                            elif inner_type is float:
                                param_schema["items"] = {"type": "number"}
                            elif inner_type is bool:
                                param_schema["items"] = {"type": "boolean"}
                    elif param.annotation is dict:
                        param_schema["type"] = "object"
                    else:
                        param_schema["type"] = "string"
                else:
                    param_schema["type"] = "string"

                properties[param_name] = param_schema

                # Check if parameter is required (no default value)
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": doc.strip().split("\n")[0],  # First line of docstring
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )

        return definitions

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get the current message history.

        Returns:
            List of message dictionaries
        """
        return list(self._messages)


class DefaultTau2AgentAdapter(AgentAdapter):
    """AgentAdapter wrapper for DefaultTau2Agent.

    Provides the standard MASEval AgentAdapter interface for DefaultTau2Agent.
    """

    def __init__(self, agent: DefaultTau2Agent, name: str = "default_agent"):
        """Initialize the adapter.

        Args:
            agent: DefaultTau2Agent instance to wrap
            name: Name for the agent adapter
        """
        super().__init__(agent, name)
        self._agent = agent

    def _run_agent(self, query: str) -> str:
        """Execute the agent with a query.

        Args:
            query: User query string

        Returns:
            Agent's response string
        """
        return self._agent.run(query)

    def get_messages(self) -> Any:
        """Get the agent's message history.

        Returns:
            Message history from the underlying agent
        """
        return self._agent.get_messages()

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this agent.

        Overrides base implementation to handle list-based message history.
        """
        history = self.get_messages()
        # history is already a list, not a MessageHistory object
        messages = history if isinstance(history, list) else []
        return {
            "type": type(self).__name__,
            "gathered_at": __import__("datetime").datetime.now().isoformat(),
            "name": self.name,
            "agent_type": type(self.agent).__name__,
            "adapter_type": type(self).__name__,
            "message_count": len(messages),
            "messages": messages,
            "callbacks": [type(cb).__name__ for cb in self.callbacks],
            "logs": self.logs,
        }


class DefaultAgentTau2Benchmark(Tau2Benchmark):
    """Tau2 benchmark with default agent implementation.

    This benchmark uses the DefaultTau2Agent which mirrors the original
    tau2-bench LLMAgent implementation for direct comparison.

    Configuration via agent_data:
        - model_id: LLM model identifier (required)
        - llm_args: Optional dict of additional LLM arguments
        - max_tool_calls: Maximum tool calls per turn (default: 50)

    Example:
        from maseval.benchmark.tau2 import DefaultAgentTau2Benchmark, load_tasks, configure_model_ids

        tasks = load_tasks("retail", split="base", limit=5)
        configure_model_ids(tasks, user_model_id="gpt-4o")

        benchmark = DefaultAgentTau2Benchmark(
            agent_data={"model_id": "gpt-4o"},
        )
        results = benchmark.run(tasks)
    """

    def __init__(
        self,
        agent_data: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
        n_task_repeats: int = 1,
        max_invocations: int = 10,
        data_dir: Optional[Path] = None,
        **kwargs: Any,
    ):
        """Initialize the default agent benchmark.

        Args:
            agent_data: Agent configuration containing:
                - model_id: LLM model identifier (required)
                - llm_args: Optional dict of additional LLM arguments
                - max_tool_calls: Maximum tool calls per turn (default: 50)
            callbacks: Benchmark callbacks
            n_task_repeats: Repetitions per task
            max_invocations: Maximum agent-user interaction rounds
            data_dir: Base data directory for domain data
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(agent_data, callbacks, n_task_repeats, max_invocations, data_dir, **kwargs)
        self._model_cache: Dict[str, ModelAdapter] = {}

    def _get_agent_model_id(self, agent_data: Dict[str, Any]) -> str:
        """Get agent model ID from agent_data.

        Args:
            agent_data: Agent configuration dict

        Returns:
            Model ID string

        Raises:
            ValueError: If model_id not configured
        """
        model_id = agent_data.get("model_id")
        if model_id is None:
            raise ValueError(
                "Agent model_id not configured in agent_data.\n"
                "Pass model_id when creating the benchmark:\n\n"
                "    benchmark = DefaultAgentTau2Benchmark(\n"
                "        agent_data={'model_id': 'gpt-4o'},\n"
                "    )"
            )
        return model_id

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create the default tau2 agent.

        Args:
            agent_data: Agent configuration with model_id
            environment: Tau2Environment with real tools
            task: Current task
            user: Optional user simulator

        Returns:
            Tuple of (agent list, agent dict)
        """
        # Get configuration
        model_id = self._get_agent_model_id(agent_data)
        llm_args = agent_data.get("llm_args", {})
        max_tool_calls = agent_data.get("max_tool_calls", 50)

        # Get tools and policy from environment
        tools = environment.create_tools()
        policy = environment.policy

        # Create model adapter
        model = self.get_model_adapter(model_id, register_name="agent_model")

        # Create agent
        agent = DefaultTau2Agent(
            tools=tools,
            policy=policy,
            model=model,
            llm_args=llm_args,
            max_tool_calls=max_tool_calls,
        )

        # Wrap in adapter
        adapter = DefaultTau2AgentAdapter(agent, name="default_agent")

        return [adapter], {"default_agent": adapter}

    @abstractmethod
    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        """Get or create a model adapter.

        Must be implemented by subclass to provide the actual ModelAdapter
        implementation for the desired LLM provider.

        Args:
            model_id: Model identifier
            **kwargs: Additional arguments (e.g., register_name for tracing)

        Returns:
            ModelAdapter instance
        """
        pass
