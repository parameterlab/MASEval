from abc import ABC
from typing import Any, Dict, List, TYPE_CHECKING

from .tracing import TraceableMixin

if TYPE_CHECKING:
    from .benchmark import Benchmark
    from .task import Task
    from .environment import Environment
    from .agent import AgentAdapter


class BenchmarkCallback(ABC, TraceableMixin):
    """Base class for benchmark callbacks."""

    def on_event(self, event_name: str, **data) -> None:
        """Handle a generic event."""
        print(f"BenchmarkCallback received event: {event_name} with data: {data}")

    def on_run_start(self, benchmark: "Benchmark"):
        pass

    def on_run_end(self, benchmark: "Benchmark", results: List[Dict]):
        pass

    def on_task_start(self, benchmark: "Benchmark", task: "Task"):
        pass

    def on_task_end(self, benchmark: "Benchmark", task: "Task", result: Dict):
        # TODO: only gets the last result if n_task_repeats > 1.
        pass

    def on_task_repeat_start(self, benchmark: "Benchmark", task: "Task", repeat_idx: int):
        pass

    def on_task_repeat_end(self, benchmark: "Benchmark", report: Dict):
        pass

    def gather_traces(self) -> dict[str, Any]:
        """Gather execution traces from this callback.

        By default, callbacks don't store traces, but subclasses can override
        this to provide custom tracing data.

        Returns:
            Dictionary with basic callback information. Subclasses should
            extend this with their own data.
        """
        return super().gather_traces()


class EnvironmentCallback(ABC, TraceableMixin):
    """Base class for environment callbacks."""

    def on_event(self, event_name: str, **data) -> None:
        """Handle a generic event."""
        print(f"EnvironmentCallback received event: {event_name} with data: {data}")

    def on_setup_start(self, environment: "Environment"):
        pass

    def on_setup_end(self, environment: "Environment"):
        pass

    def gather_traces(self) -> dict[str, Any]:
        """Gather execution traces from this callback.

        By default, callbacks don't store traces, but subclasses can override
        this to provide custom tracing data.

        Returns:
            Dictionary with basic callback information. Subclasses should
            extend this with their own data.
        """
        return super().gather_traces()


class AgentCallback(ABC, TraceableMixin):
    """Base class for agent callbacks."""

    def on_event(self, event_name: str, **data) -> None:
        """Handle a generic event."""
        print(f"AgentCallback received event: {event_name} with data: {data}")

    def on_run_start(self, agent: "AgentAdapter"):
        pass

    def on_run_end(self, agent: "AgentAdapter", result: Any):
        pass

    def gather_traces(self) -> dict[str, Any]:
        """Gather execution traces from this callback.

        By default, callbacks don't store traces, but subclasses can override
        this to provide custom tracing data.

        Returns:
            Dictionary with basic callback information. Subclasses should
            extend this with their own data.
        """
        return super().gather_traces()
