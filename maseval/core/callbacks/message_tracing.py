"""Message tracing callback for capturing agent conversations.

This callback automatically traces message history from agent runs for frameworks
that don't natively provide message history or for custom tracing needs.
"""

from typing import Any, Dict, List

from ..callback import AgentCallback
from ..agent import AgentAdapter


class MessageTracingAgentCallback(AgentCallback):
    """Callback that traces all agent messages to memory.

    This callback is useful for:
    - Frameworks that don't provide built-in message history
    - Debugging agent behavior
    - Creating datasets from agent runs
    - Monitoring multi-agent systems

    The callback collects all message history from agents after each run.

    Example:
        ```python
        from maseval import AgentAdapter
        from maseval.core.callbacks.message_tracing import MessageTracingAgentCallback

        # Create callback
        tracer = MessageTracingAgentCallback(include_metadata=True, verbose=True)

        # Use with agent
        agent_adapter = MyAgentAdapter(agent, name="agent1", callbacks=[tracer])
        agent_adapter.run("What's the weather?")

        # Access traced conversations
        for conversation in tracer.get_all_conversations():
            print(f"Agent: {conversation['agent_name']}")
            print(f"Query: {conversation['query']}")
            print(f"Messages: {len(conversation['messages'])}")
        ```
    """

    def __init__(
        self,
        include_metadata: bool = True,
        verbose: bool = False,
    ):
        """Initialize the message tracing callback.

        Args:
            include_metadata: If True, include timestamps and metadata in traces
            verbose: If True, print tracing information to console
        """
        self.include_metadata = include_metadata
        self.verbose = verbose

        # In-memory storage
        self._conversations: List[Dict[str, Any]] = []

    def on_run_start(self, agent: AgentAdapter) -> None:
        """Called when agent execution starts.

        Note: We don't have access to the query here in the current implementation,
        so we'll capture it in on_run_end from the result.
        """
        if self.verbose:
            print(f"[MessageTracingCallback] Agent '{agent.name}' starting execution")

    def on_run_end(self, agent: AgentAdapter, result: Any) -> None:
        """Called when agent execution completes.

        Args:
            agent: The agent adapter instance
            result: The result returned by the agent (usually MessageHistory)
        """
        # Get message history from agent
        history = agent.get_messages()

        # Extract query from first user message if available
        query = None
        if history and len(history) > 0:
            first_msg = history[0]
            if first_msg.get("role") == "user":
                query = first_msg.get("content", "")

        # Build conversation record (even if empty)
        conversation = {
            "agent_name": agent.name,
            "agent_type": type(agent.agent).__name__,
            "query": query,
            "messages": self._format_messages(history.to_list() if history else []),
            "message_count": len(history) if history else 0,
        }

        # Add metadata if requested
        if self.include_metadata:
            conversation["metadata"] = {
                "timestamp": history[-1].get("timestamp") if history and len(history) > 0 else None,
                "has_tool_calls": any("tool_calls" in msg for msg in history) if history else False,
                "roles": list({msg.get("role") for msg in history}) if history else [],
            }

        # Store in memory
        self._conversations.append(conversation)

        if self.verbose:
            msg_count = len(history) if history else 0
            print(f"[MessageTracingCallback] Traced {msg_count} messages from agent '{agent.name}'")

        if self.verbose:
            print(f"[MessageTracingCallback] Traced {len(history)} messages from agent '{agent.name}'")

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for tracing.

        Args:
            messages: Raw messages from MessageHistory

        Returns:
            Formatted messages (with or without metadata)
        """
        if self.include_metadata:
            return messages  # Include everything

        # Strip metadata and timestamps
        formatted = []
        for msg in messages:
            formatted_msg = {
                "role": msg.get("role"),
                "content": msg.get("content"),
            }

            # Keep essential fields
            if "name" in msg:
                formatted_msg["name"] = msg["name"]
            if "tool_calls" in msg:
                formatted_msg["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                formatted_msg["tool_call_id"] = msg["tool_call_id"]

            formatted.append(formatted_msg)

        return formatted

    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get all traced conversations from memory.

        Returns:
            List of conversation dictionaries
        """
        return self._conversations

    def get_conversations_by_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all conversations for a specific agent.

        Args:
            agent_name: Name of the agent to filter by

        Returns:
            List of conversation dictionaries for the specified agent
        """
        return [conv for conv in self._conversations if conv["agent_name"] == agent_name]

    def clear(self) -> None:
        """Clear all conversations from memory."""
        self._conversations = []
        if self.verbose:
            print("[MessageTracingCallback] Cleared all conversations from memory")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about traced conversations.

        Returns:
            Dictionary with statistics
        """
        if not self._conversations:
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "agents": [],
            }

        total_messages = sum(conv["message_count"] for conv in self._conversations)
        agents = list({conv["agent_name"] for conv in self._conversations})

        agent_stats = {}
        for agent_name in agents:
            agent_convs = self.get_conversations_by_agent(agent_name)
            agent_stats[agent_name] = {
                "conversations": len(agent_convs),
                "total_messages": sum(conv["message_count"] for conv in agent_convs),
                "avg_messages": sum(conv["message_count"] for conv in agent_convs) / len(agent_convs) if agent_convs else 0,
            }

        return {
            "total_conversations": len(self._conversations),
            "total_messages": total_messages,
            "avg_messages_per_conversation": total_messages / len(self._conversations),
            "agents": agents,
            "agent_statistics": agent_stats,
        }

    def __repr__(self) -> str:
        return f"MessageTracingAgentCallback(conversations={len(self._conversations)})"
