"""Smolagents integration for MASEval.

This module requires smolagents to be installed:
    pip install maseval[smolagents]
"""

from typing import TYPE_CHECKING, Any

from maseval import AgentAdapter, MessageHistory, User

__all__ = ["SmolAgentAdapter", "SmolAgentUser"]

# Only import smolagents types for type checking, not at runtime
if TYPE_CHECKING:
    from smolagents import UserInputTool
else:
    # Lazy import with helpful error message if smolagents is not installed
    UserInputTool = None


def _check_smolagents_installed():
    """Check if smolagents is installed and raise a helpful error if not."""
    try:
        import smolagents  # noqa: F401
    except ImportError as e:
        raise ImportError("smolagents is not installed. Install it with: pip install maseval[smolagents]") from e


class SmolAgentAdapter(AgentAdapter):
    """An AgentAdapter for smol-agents MultiStepAgent.

    Requires smolagents to be installed.

    This wrapper converts smolagents' internal message format to MASEval's
    OpenAI-compatible MessageHistory format. It automatically tracks tool calls,
    tool responses, and agent reasoning.

    Example:
        ```python
        from maseval.interface.smolagents import SmolAgentAdapter
        from smolagents import MultiStepAgent

        agent = MultiStepAgent(...)
        wrapper = SmolAgentAdapter(agent)
        result = wrapper.run("What's the weather?")

        # Access message history
        for msg in wrapper.get_messages():
            print(msg['role'], msg['content'])
        ```
    """

    def gather_traces(self) -> dict:
        """Gather traces including message history and monitoring data.

        Extends the base class to include smolagents' built-in monitoring data:
        - Token usage (input, output, total) per step and aggregated
        - Timing/duration per step and aggregated
        - Step-level details including actions and observations

        Returns:
            Dict containing messages and monitoring statistics
        """
        base_logs = super().gather_traces()
        _check_smolagents_installed()

        # Extract monitoring data from agent's memory steps
        if hasattr(self.agent, "memory") and hasattr(self.agent.memory, "steps"):
            steps_stats = []
            total_input_tokens = 0
            total_output_tokens = 0
            total_duration = 0.0

            # Import ActionStep for type checking
            from smolagents.memory import ActionStep, PlanningStep

            for step in self.agent.memory.steps:
                # Process ActionStep and PlanningStep (both have token_usage and timing)
                if isinstance(step, (ActionStep, PlanningStep)):
                    step_info = {
                        "step_number": getattr(step, "step_number", None),
                    }

                    # Add timing information
                    if hasattr(step, "timing") and step.timing:
                        step_info["duration_seconds"] = step.timing.duration
                        if step.timing.duration is not None:
                            total_duration += step.timing.duration

                    # Add token usage information
                    if hasattr(step, "token_usage") and step.token_usage:
                        step_info["input_tokens"] = step.token_usage.input_tokens
                        step_info["output_tokens"] = step.token_usage.output_tokens
                        step_info["total_tokens"] = step.token_usage.total_tokens
                        total_input_tokens += step.token_usage.input_tokens
                        total_output_tokens += step.token_usage.output_tokens

                    # Add action details for ActionStep
                    if isinstance(step, ActionStep):
                        if hasattr(step, "observations") and step.observations:
                            step_info["observations"] = step.observations
                        if hasattr(step, "action_output") and step.action_output:
                            step_info["action_output"] = str(step.action_output)
                        if hasattr(step, "error") and step.error:
                            step_info["error"] = str(step.error)

                    # Add plan for PlanningStep
                    elif isinstance(step, PlanningStep):
                        if hasattr(step, "plan") and step.plan:
                            step_info["plan"] = step.plan

                    steps_stats.append(step_info)

            # Add aggregated statistics
            base_logs.update(
                {
                    "total_steps": len(steps_stats),
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens,
                    "total_duration_seconds": total_duration,
                    "steps_detail": steps_stats,
                }
            )

        return base_logs

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this SmolAgent.

        Integrates with smolagents' native configuration system by accessing
        the agent's to_dict() method which includes comprehensive config data.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - name: Agent name
            - agent_type: Underlying agent class name
            - wrapper_type: SmolAgentAdapter
            - callbacks: List of callback class names
            - smolagents_config: Full configuration from agent.to_dict() including:
                - model: Model configuration with class and parameters
                - tools: List of tool configurations
                - max_steps: Maximum number of steps
                - planning_interval: Planning interval (if set)
                - verbosity_level: Logging verbosity
                - additional_authorized_imports: Additional imports (CodeAgent only)
                - executor_type: Code executor type (CodeAgent only)
                - managed_agents: List of managed agent configs (if any)
        """
        base_config = super().gather_config()
        _check_smolagents_installed()

        # Get comprehensive config from smolagents' native to_dict() method
        smolagents_config = {}
        if hasattr(self.agent, "to_dict"):
            try:
                smolagents_config = self.agent.to_dict()
            except Exception:
                # If to_dict fails, fall back to basic attributes
                pass

        # Add smolagents-specific config if available
        if smolagents_config:
            base_config["smolagents_config"] = smolagents_config
        else:
            # Fallback: manually collect common attributes
            config_attrs = {}
            for attr in ["max_steps", "planning_interval", "name", "description"]:
                if hasattr(self.agent, attr):
                    config_attrs[attr] = getattr(self.agent, attr)

            # CodeAgent-specific attributes
            for attr in ["additional_authorized_imports", "authorized_imports", "executor_type"]:
                if hasattr(self.agent, attr):
                    config_attrs[attr] = getattr(self.agent, attr)

            if config_attrs:
                base_config["smolagents_config"] = config_attrs

        return base_config

    def get_messages(self) -> MessageHistory:
        """Get message history by converting from smolagents memory.

        This method dynamically fetches messages from the agent's internal memory
        and converts them to MASEval format.

        Returns:
            MessageHistory with converted messages from smolagents
        """
        _check_smolagents_installed()

        # Get messages from smolagents memory
        smol_messages = self.agent.write_memory_to_messages()

        # Convert and return
        return self._convert_smolagents_messages(smol_messages)

    def set_message_history(self, history: MessageHistory) -> None:
        """Set message history - NOT SUPPORTED by smolagents.

        Args:
            history: MASEval MessageHistory to set

        Raises:
            NotImplementedError: smolagents doesn't support arbitrary message injection
        """
        raise NotImplementedError(
            "smolagents doesn't support setting arbitrary message history. "
            "The agent's memory is built from execution steps and cannot be directly manipulated. "
            "Use clear_message_history() to reset, then run() to generate new conversation."
        )

    def clear_message_history(self) -> None:
        """Clear message history by resetting smolagents memory."""
        _check_smolagents_installed()
        from smolagents.memory import AgentMemory

        # Get system prompt before clearing
        system_prompt = ""
        if hasattr(self.agent, "memory") and hasattr(self.agent.memory, "system_prompt"):
            system_prompt = self.agent.memory.system_prompt

        # Reset memory
        self.agent.memory = AgentMemory(system_prompt=system_prompt)

        # Also clear base class cache
        super().clear_message_history()

    def append_to_message_history(self, role: str, content: Any, **kwargs) -> None:
        """Append message to history - NOT SUPPORTED by smolagents.

        Args:
            role: Message role
            content: Message content (string or list)
            **kwargs: Additional message fields

        Raises:
            NotImplementedError: smolagents doesn't support arbitrary message injection
        """
        raise NotImplementedError(
            "smolagents doesn't support appending arbitrary messages to history. "
            "The agent's memory is built from execution steps and cannot be directly manipulated. "
            "Use run() to generate conversation messages."
        )

    def _run_agent(self, query: str) -> str:
        _check_smolagents_installed()

        # Run the agent (this updates the agent's internal memory and returns the final answer)
        final_answer = self.agent.run(query)

        # Return the final answer (traces are captured via get_messages())
        return final_answer

    def _convert_smolagents_messages(self, smol_messages: list) -> MessageHistory:
        """Convert smolagents message format to MASEval MessageHistory.

        Smolagents uses ChatMessage objects with MessageRole enums. This method
        normalizes them to OpenAI-compatible format with string literal roles
        while preserving tool call information.

        Args:
            smol_messages: List of ChatMessage objects from smolagents

        Returns:
            MessageHistory with converted messages
        """
        converted_messages = []

        for msg in smol_messages:
            # smolagents messages are ChatMessage objects with role and content attributes
            # Handle both dict format (from dict conversion) and ChatMessage objects
            if isinstance(msg, dict):
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
            else:
                # ChatMessage object with MessageRole enum
                role = getattr(msg, "role", "assistant")
                content = getattr(msg, "content", "")

            # Convert MessageRole enum to string literal if needed
            if hasattr(role, "value"):
                # It's an enum, extract the string value
                role = role.value  # type: ignore
            elif not isinstance(role, str):
                # Convert to string if it's something else
                role = str(role).lower()

            # Build the converted message
            converted_msg = {
                "role": role,
                "content": content,
            }

            # Handle tool calls if present
            if isinstance(msg, dict):
                if "tool_calls" in msg:
                    converted_msg["tool_calls"] = msg["tool_calls"]
                if "tool_call_id" in msg:
                    converted_msg["tool_call_id"] = msg["tool_call_id"]
                if role == "tool" and "name" in msg:
                    converted_msg["name"] = msg["name"]
                if "metadata" in msg:
                    converted_msg["metadata"] = msg["metadata"]
            else:
                # ChatMessage object - check for attributes
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    converted_msg["tool_calls"] = msg.tool_calls
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    converted_msg["tool_call_id"] = msg.tool_call_id
                if role == "tool" and hasattr(msg, "name") and msg.name:
                    converted_msg["name"] = msg.name

            converted_messages.append(converted_msg)

        return MessageHistory(converted_messages)


def _create_user_simulation_tool(user: "SmolAgentUser"):
    """Factory function to create SmolAgentUserSimulationInputTool dynamically.

    This allows us to lazily import UserInputTool and create a proper subclass.
    """
    _check_smolagents_installed()
    from smolagents import UserInputTool

    class SmolAgentUserSimulationInputTool(UserInputTool):
        """A tool that simulates user input for smolagent using the UserLLMSimulator."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._user = user

        def forward(self, question: str) -> str:
            """Simulates asking the user a question and getting a response from the user object."""
            return self._user.simulate_response(question)

    return SmolAgentUserSimulationInputTool()


class SmolAgentUser(User):
    """A smol-agent specific user that provides a tool for user interaction.

    Requires smolagents to be installed.

    Example:
        ```python
        from maseval.interface.smolagents import SmolAgentUser

        user = SmolAgentUser(...)
        tool = user.get_tool()  # Returns a SmolAgentUserSimulationInputTool
        ```
    """

    def get_tool(self):
        """Get the tool for the user."""
        return _create_user_simulation_tool(user=self)
