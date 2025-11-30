from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .environment import Environment
from .task import Task
from .user import User
from .history import MessageHistory


class Evaluator(ABC):
    """Base class for evaluators that assess task outcomes."""

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        pass

    @abstractmethod
    def __call__(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        # trace[0] is the instruction
        # user is the userclass with user specific data
        pass
