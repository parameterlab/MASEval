from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, overload
from uuid import UUID, uuid4
from collections.abc import Sequence
from typing import Iterable, List, Union, Iterator, Optional
import json
from pathlib import Path
from enum import Enum


class TimeoutAction(Enum):
    """Action to take when a task timeout occurs."""

    SKIP = "skip"  # Mark as timed out, continue to next task
    RETRY = "retry"  # Retry once with same timeout
    EXTEND = "extend"  # Double timeout and retry


@dataclass
class TaskProtocol:
    """Configuration for how MASEval executes a task.

    This is a data container for execution parameters, separate from
    task content (query, environment_data, etc.). It controls the
    interface between the task and MASEval's execution engine.

    Attributes:
        timeout_seconds: Maximum execution time for this task. None means no timeout.
        timeout_action: Action to take when timeout occurs.
        max_retries: Maximum retry attempts for transient failures (not timeouts).
        priority: Execution priority (higher = sooner). Used by adaptive task queues.
        tags: Arbitrary tags for filtering or grouping tasks.
    """

    timeout_seconds: Optional[float] = None
    timeout_action: TimeoutAction = TimeoutAction.SKIP
    max_retries: int = 0
    priority: int = 0
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A data container for a single benchmark task.

    Attributes:
        query: The main input query or prompt for the task.
        id: A unique identifier for the task (auto-generated if not provided).
        environment_data: A dictionary of data needed to set up the environment for the task.
        evaluation_data: A dictionary of data needed to evaluate the agent's performance on the task.
        metadata: A dictionary for any additional metadata about the task.
        protocol: Execution protocol controlling timeout, retries, priority, etc.
    """

    query: str
    id: UUID = field(default_factory=uuid4)
    environment_data: Dict[str, Any] = field(default_factory=dict)
    user_data: Dict[str, Any] = field(default_factory=dict)
    evaluation_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    protocol: TaskProtocol = field(default_factory=TaskProtocol)


# =============================================================================
# Task Queue Classes
# =============================================================================


class BaseTaskQueue(ABC, Sequence):
    """Abstract base class for task scheduling strategies.

    BaseTaskQueue provides a sequence-like interface for task execution with hooks
    for adaptive behavior based on task results. Concrete implementations can
    reorder tasks, skip tasks, or terminate early based on execution outcomes.

    The queue yields Task objects for execution. After each task completes,
    ``on_task_complete()`` is called with the result, allowing the queue to
    adapt its scheduling strategy.

    Subclasses must implement ``__iter__`` to define the iteration order.

    Attributes:
        _tasks: Internal list of tasks.

    Example:
        ```python
        queue = SequentialTaskQueue(tasks)

        for task in queue:
            report = execute_task(task)
            queue.on_task_complete(task, report)

            if not queue.should_continue():
                break
        ```
    """

    def __init__(self, tasks: Iterable[Task]) -> None:
        """Initialize the task queue.

        Args:
            tasks: An iterable of Task objects to schedule.
        """
        self._tasks: List[Task] = list(tasks)

    def __len__(self) -> int:
        """Return the total number of tasks in the queue."""
        return len(self._tasks)

    @overload
    def __getitem__(self, idx: int) -> Task: ...

    @overload
    def __getitem__(self, idx: slice) -> "BaseTaskQueue": ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[Task, "BaseTaskQueue"]:
        """Get a task by index or a slice of tasks.

        Args:
            idx: Integer index or slice object.

        Returns:
            A single Task for integer index, or a new queue instance for slices.
        """
        if isinstance(idx, slice):
            # Return a new instance of the same type with sliced tasks
            return self.__class__(self._tasks[idx])
        return self._tasks[idx]

    @abstractmethod
    def __iter__(self) -> Iterator[Task]:
        """Yield tasks in the scheduled execution order.

        Returns:
            Iterator yielding Task objects.
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

        Default implementation returns True. Override for early termination
        conditions (e.g., confidence threshold reached, maximum tasks processed,
        or error limit exceeded).

        Returns:
            True to continue processing, False to stop.
        """
        return True

    def append(self, task: Task) -> None:
        """Add a task to the end of the queue.

        Args:
            task: The task to append.
        """
        self._tasks.append(task)

    def extend(self, tasks: Iterable[Task]) -> None:
        """Add multiple tasks to the end of the queue.

        Args:
            tasks: An iterable of tasks to append.
        """
        self._tasks.extend(tasks)

    def to_list(self) -> List[Task]:
        """Return a copy of the internal task list.

        Returns:
            List of all tasks in the queue.
        """
        return list(self._tasks)

    @classmethod
    def from_list(cls, data: Iterable[Union[Task, dict]]) -> "BaseTaskQueue":
        """Create a queue from an iterable of Tasks or dicts.

        Args:
            data: An iterable of Task objects or dicts that can be converted to Tasks.

        Returns:
            A new queue instance containing the tasks.

        Raises:
            TypeError: If an item is neither a Task nor a dict.
        """
        tasks: List[Task] = []
        for item in data:
            if isinstance(item, Task):
                tasks.append(item)
            elif isinstance(item, dict):
                if "query" in item:
                    query = item["query"]
                    tasks.append(
                        Task(
                            query=query,
                            environment_data=item.get("environment_data", {}),
                            evaluation_data=item.get("evaluation_data", {}),
                            metadata=item.get("metadata", {}),
                        )
                    )
                else:
                    query = item.get("question") or item.get("prompt") or item.get("query") or ""
                    environment_data = (
                        item.get("environment_data") or {"text_content": item.get("text")}
                        if item.get("text")
                        else item.get("environment_data", {})
                    )
                    evaluation_data = (
                        item.get("evaluation_data") or {"short_answer": item.get("short_answer")}
                        if item.get("short_answer")
                        else item.get("evaluation_data", {})
                    )
                    tasks.append(
                        Task(
                            query=query,
                            environment_data=environment_data or {},
                            evaluation_data=evaluation_data or {},
                            metadata=item.get("metadata", {}),
                        )
                    )
            else:
                raise TypeError(f"{cls.__name__}.from_list expects Task or dict entries")
        return cls(tasks)

    @classmethod
    def from_json_file(cls, path: Union[str, Path], *, limit: Optional[int] = None) -> "BaseTaskQueue":
        """Load tasks from a JSON file.

        This helper understands the example file format used in ``examples/data.json``
        where the top-level object has a ``data`` list and optional ``metadata``.

        Args:
            path: Path to the JSON file.
            limit: Optional limit to the number of tasks to load.

        Returns:
            A new queue instance containing the loaded tasks.
        """
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        items = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload

        if limit is not None:
            items = items[:limit]

        return cls.from_list(items)


class SequentialTaskQueue(BaseTaskQueue):
    """Execute tasks in their original order.

    This queue maintains the current sequential execution model, processing
    tasks in the order they appear in the input iterable. It's the default
    queue used when no explicit queue is provided.

    Example:
        ```python
        queue = SequentialTaskQueue(tasks)
        for task in queue:
            result = execute(task)
        ```
    """

    def __iter__(self) -> Iterator[Task]:
        """Yield tasks in original order."""
        return iter(self._tasks)


class PriorityTaskQueue(BaseTaskQueue):
    """Execute tasks ordered by priority.

    Tasks are sorted by ``task.protocol.priority`` at construction time.
    Higher priority values are executed first by default. Tasks with equal
    priority maintain their relative order from the original input (stable sort).

    This queue uses ``task.protocol.priority`` as the sole source of priority.
    Pre-compute priority values and assign them to tasks before creating the queue.

    Args:
        tasks: An iterable of Task objects to schedule.
        reverse: If True (default), higher priority values execute first.
            If False, lower priority values execute first.

    Example:
        ```python
        # Assign priorities based on your criteria
        for task in tasks:
            task.protocol.priority = compute_priority(task)

        # Create queue (higher priority first)
        queue = PriorityTaskQueue(tasks)

        # Or lower priority first
        queue = PriorityTaskQueue(tasks, reverse=False)
        ```
    """

    def __init__(self, tasks: Iterable[Task], reverse: bool = True) -> None:
        """Initialize priority queue with sorted tasks.

        Args:
            tasks: An iterable of Task objects to schedule.
            reverse: If True (default), higher priority values execute first.
        """
        task_list = list(tasks)
        # Stable sort by priority
        sorted_tasks = sorted(task_list, key=lambda t: t.protocol.priority, reverse=reverse)
        super().__init__(sorted_tasks)

    def __iter__(self) -> Iterator[Task]:
        """Yield tasks in priority order."""
        return iter(self._tasks)


class AdaptiveTaskQueue(BaseTaskQueue, ABC):
    """Abstract base class for adaptive task scheduling.

    AdaptiveTaskQueue enables dynamic task ordering based on execution results.
    It integrates with the benchmark callback system to receive notifications
    after each task completes, allowing the queue to update internal state and
    adjust the execution order.

    Subclasses must implement:
        - ``_select_next_task()``: Choose the next task to execute
        - ``_update_state()``: Update internal model after task completion

    The queue maintains:
        - ``_remaining``: Tasks not yet executed
        - ``_completed``: Completed tasks paired with their reports
        - ``_stop_flag``: Flag to signal early termination

    When used with ``Benchmark.run()``, the queue is automatically registered
    as a callback if it implements the ``BenchmarkCallback`` interface.

    Example:
        ```python
        class IRTTaskQueue(AdaptiveTaskQueue):
            '''Item Response Theory-based adaptive testing.'''

            def __init__(self, tasks: Iterable[Task]):
                super().__init__(tasks)
                self._ability_estimate = 0.0

            def _select_next_task(self) -> Optional[Task]:
                if not self._remaining:
                    return None
                # Select task with difficulty closest to current ability estimate
                return min(
                    self._remaining,
                    key=lambda t: abs(t.protocol.priority - self._ability_estimate)
                )

            def _update_state(self, task: Task, report: Dict[str, Any]) -> None:
                # Update ability estimate based on task result
                correct = report.get("eval", [{}])[0].get("correct", False)
                difficulty = task.protocol.priority
                self._ability_estimate += 0.5 if correct else -0.5

        queue = IRTTaskQueue(tasks)
        results = benchmark.run(queue)  # Auto-registered as callback
        ```
    """

    def __init__(self, tasks: Iterable[Task]) -> None:
        """Initialize adaptive queue.

        Args:
            tasks: An iterable of Task objects to schedule.
        """
        super().__init__(tasks)
        self._remaining: List[Task] = list(self._tasks)
        self._completed: List[Tuple[Task, Dict[str, Any]]] = []
        self._stop_flag: bool = False

    def __iter__(self) -> Iterator[Task]:
        """Yield tasks selected by the adaptive algorithm.

        Continues until ``_select_next_task()`` returns None, ``_remaining``
        is empty, or ``_stop_flag`` is set.
        """
        while self._remaining and not self._stop_flag:
            next_task = self._select_next_task()
            if next_task is not None:
                yield next_task
            else:
                break

    def on_task_complete(self, task: Task, report: Dict[str, Any]) -> None:
        """Update state based on task result.

        Moves the task from ``_remaining`` to ``_completed`` and calls
        ``_update_state()`` to let the subclass update its internal model.

        Args:
            task: The task that just completed.
            report: The execution report.
        """
        # Find and move task from remaining to completed
        for i, t in enumerate(self._remaining):
            if t.id == task.id:
                self._completed.append((self._remaining.pop(i), report))
                break

        # Let subclass update its state
        self._update_state(task, report)

    def should_continue(self) -> bool:
        """Check if we should continue based on stopping criteria.

        Returns:
            True if stop flag is not set and tasks remain, False otherwise.
        """
        return not self._stop_flag and len(self._remaining) > 0

    def stop(self) -> None:
        """Signal that no more tasks should be processed.

        Call this from ``_update_state()`` or ``_select_next_task()`` to
        trigger early termination (e.g., when confidence threshold is reached).
        """
        self._stop_flag = True

    @abstractmethod
    def _select_next_task(self) -> Optional[Task]:
        """Select the next task to execute.

        Implement this method to define your adaptive selection algorithm
        (e.g., IRT-based selection, uncertainty sampling, bandit algorithms).

        Returns:
            The next Task to execute, or None if no suitable task is available.
        """
        pass

    @abstractmethod
    def _update_state(self, task: Task, report: Dict[str, Any]) -> None:
        """Update internal state after task completion.

        Implement this method to update ability estimates, difficulty models,
        or other state used by ``_select_next_task()``.

        Args:
            task: The task that just completed.
            report: The execution report containing status and eval results.
        """
        pass


# Alias for the default queue type
TaskQueue = SequentialTaskQueue
