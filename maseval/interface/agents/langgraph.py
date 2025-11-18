"""LangGraph integration for MASEval.

This module requires langgraph to be installed:
    pip install maseval[langgraph]
"""

from typing import TYPE_CHECKING, Any

from maseval import AgentAdapter, MessageHistory, User

__all__ = ["LangGraphAgentAdapter", "LangGraphUser"]

# Only import langgraph types for type checking, not at runtime
if TYPE_CHECKING:
    from langgraph.graph import StateGraph
    from langchain_core.messages import BaseMessage
else:
    StateGraph = None
    BaseMessage = None


def _check_langgraph_installed():
    """Check if langgraph is installed and raise a helpful error if not."""
    try:
        import langgraph  # noqa: F401
    except ImportError as e:
        raise ImportError("langgraph is not installed. Install it with: pip install maseval[langgraph]") from e


class LangGraphAgentAdapter(AgentAdapter):
    """An AgentAdapter for LangGraph CompiledGraph agents.

    Requires langgraph to be installed.

    This wrapper converts LangChain/LangGraph message types to MASEval's
    OpenAI-compatible MessageHistory format. It preserves tool calls, tool
    responses, and multi-modal content.

    LangGraph graphs can be stateless or stateful (with checkpointer). This
    wrapper supports both modes:
    - Stateless: Messages from invoke() result are cached in wrapper
    - Stateful: Messages fetched from graph state if config/thread_id provided

    Example:
        ```python
        from maseval.interface.agents.langgraph import LangGraphAgentAdapter
        from langgraph.graph import StateGraph

        # Create a LangGraph graph
        graph = StateGraph(...)
        compiled_graph = graph.compile()

        wrapper = LangGraphAgentAdapter(compiled_graph, "agent_name")
        result = wrapper.run("What's the weather?")

        # Access message history
        for msg in wrapper.get_messages():
            print(msg['role'], msg['content'])
        ```
    """

    def __init__(self, agent_instance, name: str, callbacks=None, config=None):
        """Initialize the LangGraph wrapper.

        Args:
            agent_instance: Compiled LangGraph graph
            name: Agent name
            callbacks: Optional list of callbacks
            config: Optional LangGraph config dict (for stateful graphs with checkpointer)
                   Should include 'configurable': {'thread_id': '...'} for persistent state
        """
        super().__init__(agent_instance, name, callbacks)
        self._langgraph_config = config
        self._last_result = None

    def get_messages(self) -> MessageHistory:
        """Get message history from LangGraph.

        For stateful graphs (with checkpointer and thread_id), fetches from graph state.
        For stateless graphs, returns cached messages from last run.

        Returns:
            MessageHistory with converted messages
        """
        _check_langgraph_installed()

        # If we have a config with thread_id and the graph has get_state, use it
        if self._langgraph_config and hasattr(self.agent, "get_state"):
            try:
                state = self.agent.get_state(self._langgraph_config)
                messages = state.values.get("messages", [])
                if messages:
                    return self._convert_langchain_messages(messages)
            except Exception:
                # If get_state fails, fall back to cached result
                pass

        # Fall back to cached result from last run
        if self._last_result:
            messages = self._last_result.get("messages", [])
            return self._convert_langchain_messages(messages)

        # No messages available
        return MessageHistory()

    def set_message_history(self, history: MessageHistory) -> None:
        """Set message history for langgraph.

        For stateless graphs, updates the cached result.
        For stateful graphs, this is not fully supported as LangGraph manages state internally.

        Args:
            history: MASEval MessageHistory to set
        """
        _check_langgraph_installed()
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

        # Convert MessageHistory to LangChain messages
        lc_messages = []
        for msg in history:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")

            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                lc_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))

        # Update cached result
        self._last_result = {"messages": lc_messages}

        # Also update base class cache
        super().set_message_history(history)

    def clear_message_history(self) -> None:
        """Clear message history for langgraph.

        Clears the cached result. For stateful graphs, this doesn't clear
        the persistent state in the checkpointer.
        """
        self._last_result = None
        super().clear_message_history()

    def append_to_message_history(self, role: str, content: Any, **kwargs) -> None:
        """Append message to history.

        For stateless graphs, this appends to the cached result.
        For stateful graphs, messages are managed by LangGraph during invoke().

        Args:
            role: Message role
            content: Message content (string or list)
            **kwargs: Additional message fields
        """
        _check_langgraph_installed()
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        # Get current messages
        current_messages = []
        if self._last_result and "messages" in self._last_result:
            current_messages = self._last_result["messages"]

        # Create new message
        if role == "user":
            new_msg = HumanMessage(content=str(content))
        elif role == "assistant":
            new_msg = AIMessage(content=str(content))
        elif role == "system":
            new_msg = SystemMessage(content=str(content))
        else:
            new_msg = AIMessage(content=str(content))

        # Append and update cache
        current_messages.append(new_msg)
        self._last_result = {"messages": current_messages}

        # Also update base class cache
        super().append_to_message_history(role, content, **kwargs)

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this LangGraph agent.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - name: Agent name
            - agent_type: CompiledGraph or similar
            - wrapper_type: LangGraphAgentAdapter
            - callbacks: List of callback class names
            - has_checkpointer: Whether the graph has state persistence
            - config: LangGraph config dict (with sensitive data removed)
            - graph_info: Information about the graph structure (if available)
        """
        base_config = super().gather_config()
        _check_langgraph_installed()

        # Add LangGraph-specific config
        langgraph_config = {}

        # Check if graph has checkpointer (state persistence)
        if hasattr(self.agent, "checkpointer"):
            langgraph_config["has_checkpointer"] = self.agent.checkpointer is not None

        # Include config (but remove sensitive data like thread_id which might be user-specific)
        if self._langgraph_config:
            safe_config = {}
            for key, value in self._langgraph_config.items():
                if key != "configurable":  # Skip configurable section which may have thread_id
                    safe_config[key] = value
                else:
                    # Include configurable but sanitize
                    safe_config["configurable"] = {"has_thread_id": "thread_id" in value if isinstance(value, dict) else False}
            langgraph_config["config"] = safe_config

        # Try to get graph structure info
        if hasattr(self.agent, "get_graph"):
            try:
                graph = self.agent.get_graph()
                if graph:
                    langgraph_config["graph_info"] = {
                        "num_nodes": len(graph.nodes) if hasattr(graph, "nodes") else None,
                        "num_edges": len(graph.edges) if hasattr(graph, "edges") else None,
                    }
            except Exception:
                pass

        if langgraph_config:
            base_config["langgraph_config"] = langgraph_config

        return base_config

    def _run_agent(self, query: str) -> Any:
        _check_langgraph_installed()
        from langchain_core.messages import HumanMessage

        # Initialize the state with the user query
        initial_state = {"messages": [HumanMessage(content=query)]}

        # Invoke the graph (with config if provided)
        if self._langgraph_config:
            result = self.agent.invoke(initial_state, config=self._langgraph_config)
        else:
            result = self.agent.invoke(initial_state)

        # Cache the result for stateless graphs
        self._last_result = result

        # Extract and return the final answer from the graph's result
        # LangGraph typically returns dict with 'messages' key, extract the last AI message
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            # Return the content of the last message as the final answer
            return getattr(last_message, "content", str(last_message))

        return None

    def _convert_langchain_messages(self, lc_messages: list) -> MessageHistory:
        """Convert LangChain messages to MASEval MessageHistory format.

        LangChain uses typed message classes (HumanMessage, AIMessage, etc.).
        This method converts them to OpenAI-compatible dict format.

        Args:
            lc_messages: List of LangChain message objects

        Returns:
            MessageHistory with converted messages
        """
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

        history_messages = []

        for msg in lc_messages:
            if isinstance(msg, HumanMessage):
                history_messages.append(
                    {
                        "role": "user",
                        "content": msg.content,
                    }
                )
            elif isinstance(msg, AIMessage):
                message_dict = {
                    "role": "assistant",
                    "content": msg.content if msg.content else "",
                }
                # Include tool calls if present
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    message_dict["tool_calls"] = tool_calls
                history_messages.append(message_dict)
            elif isinstance(msg, ToolMessage):
                message_dict = {
                    "role": "tool",
                    "content": msg.content,
                }
                # Include tool metadata
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id:
                    message_dict["tool_call_id"] = tool_call_id
                tool_name = getattr(msg, "name", None)
                if tool_name:
                    message_dict["name"] = tool_name
                history_messages.append(message_dict)
            elif isinstance(msg, SystemMessage):
                history_messages.append(
                    {
                        "role": "system",
                        "content": msg.content,
                    }
                )

        return MessageHistory(history_messages)


class LangGraphUser(User):
    """A LangGraph-specific user that provides a tool for user interaction.

    Requires langgraph to be installed.

    Example:
        ```python
        from maseval.interface.agents.langgraph import LangGraphUser

        user = LangGraphUser(...)
        tool = user.get_tool()  # Returns a LangChain tool
        ```
    """

    def get_tool(self):
        """Get a LangChain-compatible tool for user interaction."""
        _check_langgraph_installed()
        from langchain_core.tools import tool

        user_instance = self

        @tool
        def ask_user(question: str) -> str:
            """Ask the user a question and get their response.

            Args:
                question: The question to ask the user

            Returns:
                The user's response
            """
            return user_instance.simulate_response(question)

        return ask_user
