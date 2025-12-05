"""Top-level package exports for convenience.

Expose a small, stable surface area for users to import core abstractions directly from `maseval`,
for example: `from maseval import Task, Benchmark`.

Core library sits in the top namespace for easy access.
Interfaces sit in the `maseval.interface` submodule.
Benchmarks sit in the `maseval.benchmark` submodule.
"""

from .core.task import Task, TaskCollection
from .core.environment import Environment
from .core.agent import AgentAdapter
from .core.benchmark import Benchmark, TaskExecutionStatus
from .core.callback_handler import CallbackHandler
from .core.callback import BenchmarkCallback, EnvironmentCallback, AgentCallback
from .core.callbacks import MessageTracingAgentCallback
from .core.simulator import ToolLLMSimulator, UserLLMSimulator, SimulatorError
from .core.model import ModelAdapter
from .core.user import User, TerminationReason
from .core.evaluator import Evaluator
from .core.history import MessageHistory, ToolInvocationHistory
from .core.tracing import TraceableMixin

__all__ = [
    "Task",
    "TaskCollection",
    "Environment",
    "AgentAdapter",
    "Benchmark",
    "TaskExecutionStatus",
    "CallbackHandler",
    "BenchmarkCallback",
    "EnvironmentCallback",
    "AgentCallback",
    "MessageTracingAgentCallback",
    "ToolLLMSimulator",
    "UserLLMSimulator",
    "SimulatorError",
    "User",
    "TerminationReason",
    "MessageHistory",
    "Evaluator",
    "ToolInvocationHistory",
    "ModelAdapter",
    "TraceableMixin",
]
