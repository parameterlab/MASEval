"""Optional smolagents components that require smolagents to be installed.

This module contains classes that directly inherit from smolagents types.
It is imported lazily to allow the main smolagents module to work without
smolagents installed (for type checking and documentation).
"""

from typing import TYPE_CHECKING

from smolagents import UserInputTool

if TYPE_CHECKING:
    from maseval.interface.agents.smolagents import SmolAgentUser

__all__ = ["SmolAgentUserSimulationInputTool"]


class SmolAgentUserSimulationInputTool(UserInputTool):
    """A tool that simulates user input for smolagents using the User simulator.

    This class directly inherits from `smolagents.UserInputTool` and can be passed
    to any smolagent. It wraps a `SmolAgentUser` and intercepts user input requests,
    routing them through the user's LLM-based response simulation.

    Note:
        Don't instantiate this directly. Use `SmolAgentUser.get_tool()` instead.

    Example:
        ```python
        from maseval.interface.agents.smolagents import SmolAgentUser

        user = SmolAgentUser(model=model, persona="Helpful user", scenario="Book a flight")
        tool = user.get_tool()  # Returns SmolAgentUserSimulationInputTool instance

        # Pass to your smolagent
        agent = CodeAgent(tools=[tool, ...], model=model)
        ```

    Attributes:
        _user: The SmolAgentUser instance that handles response simulation.
    """

    def __init__(self, user: "SmolAgentUser"):
        """Initialize the tool with a SmolAgentUser.

        Args:
            user: The SmolAgentUser instance to wrap for response simulation.
        """
        super().__init__()
        self._user = user

    def forward(self, question: str) -> str:
        """Ask the user a question and get a response.

        This method is called by smolagents when the agent needs user input.
        It delegates to the wrapped SmolAgentUser's respond method.

        Args:
            question: The question to ask the user.

        Returns:
            The user's response.
        """
        return self._user.respond(question)
