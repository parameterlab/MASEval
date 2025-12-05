"""Task queue abstraction for flexible task scheduling.

This module provides the TaskQueue abstract base class and concrete implementations
for different task scheduling strategies. The queue abstraction replaces the static
`for task in tasks` loop with a dynamic scheduling system that enables:

1. Dynamic task ordering
2. Callback-driven scheduling (adaptive testing)
3. Priority-based execution
4. Conditional task skipping
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .task import Task, TaskCollection


class TaskQueue(ABC):
    """Abstract base for task scheduling strategies.

    TaskQueue provides an iterator interface for task execution with hooks
    for adaptive behavior based on task results. Concrete implementations
    can reorder tasks, skip tasks, or terminate early based on execution
    outcomes.

    The queue yields (Task, agent_data) tuples for execution. After each
    task completes, `on_task_complete()` is called with the result, allowing
    the queue to adapt its scheduling strategy.

    Usage:
        queue = SequentialQueue(tasks, agent_data_list)

        for task, agent_data in queue:
            report = execute_task(task, agent_data)
            queue.on_task_complete(task, report)

            if not queue.should_continue():
                break
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[Task, Dict[str, Any]]]:
        """Yield (task, agent_data) pairs in execution order.

        Returns:
            Iterator yielding tuples of (Task, agent_data dict).
        """
        pass

    def on_task_complete(self, task: Task, report: Dict[str, Any]) -> None:
        """Called after each task completes.

        Override this method for adaptive scheduling behavior that responds
        to task execution results (e.g., updating ability estimates, adjusting
        priorities, or marking related tasks for skipping).

        Args:
            task: The task that just completed.
            report: The execution report containing status, traces, and eval results.
        """
        pass

    def should_continue(self) -> bool:
        """Whether to continue processing tasks.

        Default implementation returns True until the queue is exhausted.
        Override for early termination conditions (e.g., confidence threshold
        reached, maximum tasks processed, or error limit exceeded).

        Returns:
            True to continue processing, False to stop.
        """
        return True


class SequentialQueue(TaskQueue):
    """Execute tasks in their original order (default behavior).

    This queue maintains the current sequential execution model, processing
    tasks in the order they appear in the task collection. It's the default
    queue used when no explicit queue is provided.

    Attributes:
        tasks: List of (Task, agent_data) pairs.
    """

    def __init__(self, tasks: TaskCollection, agent_data_list: List[Dict[str, Any]]):
        """Initialize sequential queue.

        Args:
            tasks: Collection of tasks to execute.
            agent_data_list: List of agent configuration dicts, one per task.
        """
        self._tasks: List[Tuple[Task, Dict[str, Any]]] = list(zip(tasks, agent_data_list))
        self._index = 0

    def __iter__(self) -> Iterator[Tuple[Task, Dict[str, Any]]]:
        """Yield tasks in original order."""
        for task, agent_data in self._tasks:
            yield task, agent_data


class PriorityQueue(TaskQueue):
    """Execute tasks by priority (from TaskProtocol.priority).

    Tasks with higher priority values are executed first. Tasks with equal
    priority maintain their relative order from the original collection.

    This queue is useful when some tasks are more important or time-sensitive
    than others, or when you want to process easier tasks first to get quick
    feedback.

    Attributes:
        tasks: List of (Task, agent_data) pairs sorted by priority.
    """

    def __init__(self, tasks: TaskCollection, agent_data_list: List[Dict[str, Any]]):
        """Initialize priority queue.

        Args:
            tasks: Collection of tasks to execute.
            agent_data_list: List of agent configuration dicts, one per task.
        """
        paired = list(zip(tasks, agent_data_list))
        # Sort by priority descending (higher priority first)
        # Use enumerate to maintain stable sort for equal priorities
        self._tasks: List[Tuple[Task, Dict[str, Any]]] = sorted(paired, key=lambda x: x[0].protocol.priority, reverse=True)

    def __iter__(self) -> Iterator[Tuple[Task, Dict[str, Any]]]:
        """Yield tasks in priority order."""
        for task, agent_data in self._tasks:
            yield task, agent_data


class AdaptiveQueue(TaskQueue):
    """Base class for adaptive task scheduling.

    Adaptive queues adjust task order based on execution results. This is
    useful for techniques like Item Response Theory (IRT) based testing,
    where task selection optimizes for information gain about agent ability.

    Subclasses should override `_select_next_task()` to implement their
    selection algorithm, and `_update_state()` to update internal state
    after each task completion.

    Attributes:
        pending: Tasks not yet executed.
        completed: Tasks that have been executed with their reports.
    """

    def __init__(self, tasks: TaskCollection, agent_data_list: List[Dict[str, Any]]):
        """Initialize adaptive queue.

        Args:
            tasks: Collection of tasks to execute.
            agent_data_list: List of agent configuration dicts, one per task.
        """
        self._pending: List[Tuple[Task, Dict[str, Any]]] = list(zip(tasks, agent_data_list))
        self._completed: List[Tuple[Task, Dict[str, Any]]] = []
        self._stop_flag = False

    def __iter__(self) -> Iterator[Tuple[Task, Dict[str, Any]]]:
        """Yield tasks selected by the adaptive algorithm."""
        while self._pending and not self._stop_flag:
            next_item = self._select_next_task()
            if next_item is not None:
                yield next_item
            else:
                break

    def on_task_complete(self, task: Task, report: Dict[str, Any]) -> None:
        """Update state based on task result.

        Args:
            task: The task that just completed.
            report: The execution report.
        """
        # Find and move task from pending to completed
        for i, (t, agent_data) in enumerate(self._pending):
            if t.id == task.id:
                self._completed.append(self._pending.pop(i))
                break

        # Update adaptive state
        self._update_state(task, report)

    def should_continue(self) -> bool:
        """Check if we should continue based on stopping criteria."""
        return not self._stop_flag and len(self._pending) > 0

    def stop(self) -> None:
        """Signal that no more tasks should be processed."""
        self._stop_flag = True

    def _select_next_task(self) -> Optional[Tuple[Task, Dict[str, Any]]]:
        """Select the next task to execute.

        Override this method to implement custom selection algorithms
        (e.g., IRT-based selection, uncertainty sampling, etc.).

        Default implementation returns tasks in order (first remaining task).

        Returns:
            The next (Task, agent_data) pair, or None if no suitable task.
        """
        if not self._pending:
            return None
        return self._pending[0]

    def _update_state(self, task: Task, report: Dict[str, Any]) -> None:
        """Update internal state after task completion.

        Override this method to update ability estimates, difficulty models,
        or other state used by `_select_next_task()`.

        Args:
            task: The task that just completed.
            report: The execution report containing status and eval results.
        """
        pass
