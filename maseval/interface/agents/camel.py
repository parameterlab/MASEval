"""CAMEL-AI integration for MASEval.

This module requires camel-ai to be installed:
    pip install maseval[camel]
"""

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List

from maseval import AgentAdapter, MessageHistory, User

__all__ = ["CamelAgentAdapter", "CamelUser"]

# Only import camel types for type checking, not at runtime
if TYPE_CHECKING:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
else:
    ChatAgent = None
    BaseMessage = None


def _check_camel_installed():
    """Check if camel-ai is installed and raise a helpful error if not."""
    try:
        import camel  # noqa: F401
    except ImportError as e:
        raise ImportError("camel-ai is not installed. Install it with: pip install maseval[camel]") from e


class CamelAgentAdapter(AgentAdapter):
    """An AgentAdapter for CAMEL-AI ChatAgent.

    This adapter integrates CAMEL-AI's ChatAgent with MASEval's benchmarking framework,
    converting CAMEL's message format to OpenAI-compatible MessageHistory format.
    It leverages CAMEL's native memory system as the source of truth for conversation
    history, ensuring accurate tracking of multi-turn interactions.

    CAMEL-AI is a modular framework for building intelligent multi-agent systems.
    The ChatAgent is its core component for single-agent interactions, supporting
    tool calling, memory management, and various LLM backends.

    How to use:
        1. **Create a CAMEL ChatAgent** with system message and optional tools
        2. **Wrap with CamelAgentAdapter** to enable MASEval integration
        3. **Use in benchmarks** or call directly for testing
        4. **Access traces and config** for analysis and debugging

        Example workflow:
            ```python
            from maseval.interface.agents.camel import CamelAgentAdapter
            from camel.agents import ChatAgent
            from camel.models import ModelFactory
            from camel.types import ModelPlatformType, ModelType

            # Create a CAMEL model
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
            )

            # Create a CAMEL ChatAgent
            agent = ChatAgent(
                system_message="You are a helpful assistant.",
                model=model,
            )

            # Wrap with adapter
            agent_adapter = CamelAgentAdapter(agent, name="assistant")

            # Run agent
            result = agent_adapter.run("What is the capital of France?")

            # Access message history in OpenAI format
            for msg in agent_adapter.get_messages():
                print(f"{msg['role']}: {msg['content']}")

            # Gather execution traces
            traces = agent_adapter.gather_traces()
            print(f"Messages: {traces['message_count']}")

            # Gather configuration
            config = agent_adapter.gather_config()
            print(f"Model: {config.get('camel_config', {}).get('model_type')}")

            # Use in benchmark
            benchmark = MyBenchmark(agent_data={"agent": agent_adapter})
            results = benchmark.run(tasks)
            ```

    Message Format:
        CAMEL uses BaseMessage objects with role_name, role_type, and content.
        The adapter converts these to OpenAI-compatible format via the agent's
        memory system, which already provides messages in a compatible structure.

    Memory as Source of Truth:
        Following MASEval's adapter pattern, this adapter uses CAMEL's native
        memory storage as the single source of truth. Messages are dynamically
        fetched from `agent.memory.get_context()` rather than being cached,
        ensuring consistency with the agent's internal state.

    Execution Model:
        CAMEL's ChatAgent uses a `step()` method for execution, which processes
        one turn of conversation and returns a ChatAgentResponse. The adapter
        handles this automatically, extracting the final answer from the response.

    Requires:
        camel-ai to be installed: `pip install maseval[camel]`
    """

    def __init__(self, agent_instance, name: str, callbacks=None):
        """Initialize the CAMEL adapter.

        Args:
            agent_instance: CAMEL ChatAgent instance
            name: Agent name for identification
            callbacks: Optional list of AgentCallback instances
        """
        super().__init__(agent_instance, name, callbacks)
        self._last_response = None

    def get_messages(self) -> MessageHistory:
        """Get message history from CAMEL's memory system.

        Dynamically fetches messages from the agent's memory, converting
        them to MASEval's MessageHistory format. CAMEL's memory.get_context()
        returns messages in OpenAI-compatible format.

        Returns:
            MessageHistory with converted messages
        """
        _check_camel_installed()

        messages: List[Dict[str, Any]] = []

        # Try to get messages from agent's memory
        if hasattr(self.agent, "memory") and self.agent.memory is not None:
            try:
                # get_context() returns (messages_list, token_count)
                context = self.agent.memory.get_context()
                if isinstance(context, tuple) and len(context) >= 1:
                    memory_messages = context[0]
                    if isinstance(memory_messages, list):
                        messages = self._convert_memory_messages(memory_messages)
            except Exception:
                # If memory access fails, fall back to empty
                pass

        return MessageHistory(messages)

    def _convert_memory_messages(self, memory_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert CAMEL memory messages to MASEval format.

        CAMEL's memory.get_context() typically returns messages already in
        OpenAI-compatible format with 'role' and 'content' keys.

        Args:
            memory_messages: List of message dicts from CAMEL memory

        Returns:
            List of message dicts in MASEval/OpenAI format
        """
        converted = []
        for msg in memory_messages:
            if isinstance(msg, dict):
                # Already in dict format, normalize role names
                role = msg.get("role", "assistant")
                content = msg.get("content", "")

                # Normalize role to standard values
                role_mapping = {
                    "system": "system",
                    "user": "user",
                    "assistant": "assistant",
                    "tool": "tool",
                    "function": "assistant",
                }
                normalized_role = role_mapping.get(role.lower() if isinstance(role, str) else "assistant", "assistant")

                converted_msg: Dict[str, Any] = {
                    "role": normalized_role,
                    "content": content,
                }

                # Preserve tool-related fields if present
                if "tool_calls" in msg:
                    converted_msg["tool_calls"] = msg["tool_calls"]
                if "tool_call_id" in msg:
                    converted_msg["tool_call_id"] = msg["tool_call_id"]
                if normalized_role == "tool" and "name" in msg:
                    converted_msg["name"] = msg["name"]

                converted.append(converted_msg)
            else:
                # Handle BaseMessage objects
                converted.append(self._convert_base_message(msg))

        return converted

    def _convert_base_message(self, msg) -> Dict[str, Any]:
        """Convert a CAMEL BaseMessage to MASEval format.

        Args:
            msg: CAMEL BaseMessage object

        Returns:
            Message dict in MASEval/OpenAI format
        """
        # Extract role from role_type enum or role_name
        role = "assistant"
        if hasattr(msg, "role_type"):
            role_type = msg.role_type
            if hasattr(role_type, "value"):
                role = role_type.value.lower()
            elif hasattr(role_type, "name"):
                role = role_type.name.lower()

        # Map CAMEL roles to OpenAI roles
        role_mapping = {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "critic": "assistant",  # CAMEL-specific role
            "embodiment": "assistant",  # CAMEL-specific role
            "default": "assistant",
        }
        normalized_role = role_mapping.get(role, "assistant")

        # Extract content
        content = getattr(msg, "content", "")

        return {
            "role": normalized_role,
            "content": content,
        }

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this CAMEL agent.

        Extends the base class to include CAMEL-specific execution data
        such as response metadata and termination status.

        Returns:
            Dictionary containing:
            - Base traces (type, gathered_at, name, messages, logs, etc.)
            - last_response_terminated: Whether the last response indicated termination
            - last_response_info: Additional info from the last response
        """
        base_traces = super().gather_traces()
        _check_camel_installed()

        # Add CAMEL-specific trace data
        if self._last_response is not None:
            if hasattr(self._last_response, "terminated"):
                base_traces["last_response_terminated"] = self._last_response.terminated

            if hasattr(self._last_response, "info") and self._last_response.info:
                # Include response info, filtering out non-serializable items
                try:
                    base_traces["last_response_info"] = dict(self._last_response.info)
                except (TypeError, ValueError):
                    base_traces["last_response_info"] = str(self._last_response.info)

        return base_traces

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this CAMEL agent.

        Returns:
            Dictionary containing:
            - Base config (type, gathered_at, name, agent_type, adapter_type, callbacks)
            - camel_config: CAMEL-specific configuration including:
                - system_message: The agent's system prompt
                - model_type: The model being used
                - tools: List of configured tools
                - memory_type: Type of memory being used
        """
        base_config = super().gather_config()
        _check_camel_installed()

        camel_config: Dict[str, Any] = {}

        # Get system message
        if hasattr(self.agent, "system_message"):
            sys_msg = self.agent.system_message
            if hasattr(sys_msg, "content"):
                camel_config["system_message"] = sys_msg.content
            elif isinstance(sys_msg, str):
                camel_config["system_message"] = sys_msg

        # Get model information
        if hasattr(self.agent, "model"):
            model = self.agent.model
            if hasattr(model, "model_type"):
                model_type = model.model_type
                if hasattr(model_type, "value"):
                    camel_config["model_type"] = model_type.value
                else:
                    camel_config["model_type"] = str(model_type)
            if hasattr(model, "model_config_dict"):
                camel_config["model_config"] = model.model_config_dict

        # Get tools information
        if hasattr(self.agent, "tools") and self.agent.tools:
            tools_info = []
            for tool in self.agent.tools:
                tool_info: Dict[str, Any] = {}
                if hasattr(tool, "name"):
                    tool_info["name"] = tool.name
                elif hasattr(tool, "get_function_name"):
                    tool_info["name"] = tool.get_function_name()
                if hasattr(tool, "description"):
                    tool_info["description"] = tool.description
                elif hasattr(tool, "get_function_description"):
                    tool_info["description"] = tool.get_function_description()
                if tool_info:
                    tools_info.append(tool_info)
            if tools_info:
                camel_config["tools"] = tools_info

        # Get memory type
        if hasattr(self.agent, "memory") and self.agent.memory is not None:
            camel_config["memory_type"] = type(self.agent.memory).__name__

        if camel_config:
            base_config["camel_config"] = camel_config

        return base_config

    def _run_agent(self, query: str) -> str:
        """Execute the CAMEL agent with the given query.

        Uses CAMEL's step() method to process one turn of conversation,
        extracting the final answer from the response.

        Args:
            query: The user query to send to the agent

        Returns:
            The agent's response content as a string

        Raises:
            Exception: If agent execution fails
        """
        _check_camel_installed()
        from camel.messages import BaseMessage

        start_time = time.time()
        timestamp = datetime.now().isoformat()

        try:
            # Create user message for CAMEL
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=query,
            )

            # Execute one step
            response = self.agent.step(user_msg)

            # Cache the response for traces
            self._last_response = response
            duration = time.time() - start_time

            # Log successful execution
            log_entry: Dict[str, Any] = {
                "timestamp": timestamp,
                "query": query,
                "query_length": len(query),
                "duration_seconds": duration,
                "status": "success",
            }

            # Extract response details
            if hasattr(response, "msgs") and response.msgs:
                log_entry["response_count"] = len(response.msgs)

            if hasattr(response, "terminated"):
                log_entry["terminated"] = response.terminated

            if hasattr(response, "info") and response.info:
                # Extract useful info like token usage if available
                info = response.info
                if isinstance(info, dict):
                    if "usage" in info:
                        usage = info["usage"]
                        if isinstance(usage, dict):
                            log_entry["input_tokens"] = usage.get("prompt_tokens", 0)
                            log_entry["output_tokens"] = usage.get("completion_tokens", 0)
                            log_entry["total_tokens"] = usage.get("total_tokens", 0)

            self.logs.append(log_entry)

            # Extract and return the final answer
            return self._extract_final_answer(response)

        except Exception as e:
            duration = time.time() - start_time

            # Log failed execution
            self.logs.append(
                {
                    "timestamp": timestamp,
                    "query": query,
                    "query_length": len(query),
                    "duration_seconds": duration,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

            raise

    def _extract_final_answer(self, response) -> str:
        """Extract the final answer from a CAMEL ChatAgentResponse.

        Args:
            response: ChatAgentResponse from agent.step()

        Returns:
            The assistant's response content as a string
        """
        # ChatAgentResponse has msgs attribute with response messages
        if hasattr(response, "msgs") and response.msgs:
            # Get the first (and typically only) response message
            msg = response.msgs[0]
            if hasattr(msg, "content"):
                return str(msg.content)
            return str(msg)

        # Fallback: try msg attribute (singular)
        if hasattr(response, "msg") and response.msg:
            msg = response.msg
            if hasattr(msg, "content"):
                return str(msg.content)
            return str(msg)

        # Last resort: string representation
        return str(response)


class CamelUser(User):
    """A CAMEL-specific user that provides a tool for user interaction.

    This class extends the base User class to provide a CAMEL-compatible
    FunctionTool that wraps the simulate_response method, allowing CAMEL
    agents to interact with simulated users during benchmarking.

    Requires camel-ai to be installed.

    Example:
        ```python
        from maseval.interface.agents.camel import CamelUser
        from maseval.interface.inference import OpenAIModelAdapter

        # Create a model for user simulation
        model = OpenAIModelAdapter(model_id="gpt-4o-mini")

        # Create the user
        user = CamelUser(
            name="customer",
            model=model,
            user_profile={"name": "John", "preferences": ["fast service"]},
            scenario="Customer seeking help with a product return",
            initial_query="I need to return a product I bought last week.",
        )

        # Get the tool for use with CAMEL agent
        tool = user.get_tool()

        # Create CAMEL agent with the user tool
        from camel.agents import ChatAgent
        agent = ChatAgent(
            system_message="You are a helpful customer service agent.",
            tools=[tool],
        )
        ```
    """

    def get_tool(self):
        """Get a CAMEL-compatible tool for user interaction.

        Returns a CAMEL FunctionTool that wraps the simulate_response method,
        allowing agents to ask the user questions during execution.

        Returns:
            CAMEL FunctionTool instance for user interaction
        """
        _check_camel_installed()
        from camel.toolkits import FunctionTool

        user_instance = self

        def ask_user(question: str) -> str:
            """Ask the user a question and get their response.

            Use this tool when you need to ask the user for clarification,
            additional information, or to confirm something.

            Args:
                question: The question to ask the user

            Returns:
                The user's response to the question
            """
            return user_instance.simulate_response(question)

        return FunctionTool(func=ask_user)
