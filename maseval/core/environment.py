from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .callback import EnvironmentCallback
from .tracing import TraceableMixin
from .config import ConfigurableMixin


class Environment(ABC, TraceableMixin, ConfigurableMixin):
    """Manages the state and tools available during a task execution."""

    def __init__(self, task_data: Dict[str, Any], callbacks: Optional[List[EnvironmentCallback]] = None):
        super().__init__()
        self.callbacks = callbacks or []
        for cb in self.callbacks:
            cb.on_setup_start(self)
        self.state = self.setup_state(task_data)
        self.tools = self.create_tools()
        # Store tools in a dict for easier lookup during tracing
        self._tools_dict: Dict[str, Any] = {}
        if isinstance(self.tools, list):
            for tool in self.tools:
                tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", str(type(tool).__name__))
                self._tools_dict[tool_name] = tool
        for cb in self.callbacks:
            cb.on_setup_end(self)

    @abstractmethod
    def setup_state(self, task_data: dict) -> Any:
        """Initializes the environment's state from task data."""
        pass

    @abstractmethod
    def create_tools(self) -> list:
        """Creates tools that can interact with the environment's state."""
        pass

    def get_tools(self) -> list:
        return self.tools

    def gather_traces(self) -> dict[str, Any]:
        """Gather execution traces from this environment and its tools.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - tool_count: Number of tools in environment
            - tools: Dictionary of tool traces keyed by tool name
        """
        tool_traces = {}

        for tool_name, tool in self._tools_dict.items():
            # Try to gather traces from the tool
            if hasattr(tool, "gather_traces"):
                tool_traces[tool_name] = tool.gather_traces()
            # Check for ToolInvocationHistory
            elif hasattr(tool, "history") and hasattr(tool.history, "to_list"):
                tool_traces[tool_name] = {
                    "type": type(tool).__name__,
                    "invocations": tool.history.to_list(),
                    "total_invocations": len(tool.history.to_list()),
                }
            # Check if it has a simulator (common pattern in examples)
            elif hasattr(tool, "simulator") and hasattr(tool.simulator, "gather_traces"):
                tool_traces[tool_name] = {
                    "type": type(tool).__name__,
                    "simulator_traces": tool.simulator.gather_traces(),
                }
            else:
                # Basic info if no tracing capability
                tool_traces[tool_name] = {
                    "type": type(tool).__name__,
                    "has_tracing": False,
                }

        return {
            **super().gather_traces(),
            "tool_count": len(self._tools_dict),
            "tools": tool_traces,
        }

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this environment.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - tool_count: Number of tools
            - tool_names: List of tool names
        """
        return {
            **super().gather_config(),
            "tool_count": len(self._tools_dict),
            "tool_names": list(self._tools_dict.keys()),
        }
