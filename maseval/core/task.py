from dataclasses import dataclass, field
from typing import Any, Dict
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


class TaskCollection(Sequence):
    """A lightweight, sequence-like container for `Task` objects.

    Usage:
        - Construct from an iterable of `Task` or dicts: `TaskCollection.from_list(data)`
        - Load from an examples-style JSON: `TaskCollection.from_json_file("examples/data.json")`

    The collection is immutable from the Sequence API perspective (supports indexing and slicing),
    but provides `append`/`extend` helpers for convenience when building programmatically.
    """

    def __init__(self, tasks: Optional[Iterable[Task]] = None) -> None:
        """Initialize the TaskCollection.

        Args:
            tasks: An optional iterable of `Task` objects to initialize the collection.
        """
        # TODO for any element in the iterable that is not a Task, convert it to a Task
        self._tasks: List[Task] = list(tasks) if tasks is not None else []

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, idx):
        # Return a Task for int index, or a new TaskCollection for slices (pythonic behaviour)
        if isinstance(idx, slice):
            return TaskCollection(self._tasks[idx])
        return self._tasks[idx]

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TaskCollection({len(self._tasks)} tasks)"

    # Convenience mutators
    def append(self, task: Task) -> None:
        self._tasks.append(task)

    def extend(self, tasks: Iterable[Task]) -> None:
        self._tasks.extend(tasks)

    def to_list(self) -> List[Task]:
        return list(self._tasks)

    # Factories
    @classmethod
    def from_list(cls, data: Iterable[Union[Task, dict]]) -> "TaskCollection":
        tasks: List[Task] = []
        for item in data:
            if isinstance(item, Task):
                tasks.append(item)
            elif isinstance(item, dict):
                # Expect a dict that can be turned into a Task
                # Accept both full Task kwargs or the lightweight example format
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
                    # Attempt to map common example keys
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
                raise TypeError("TaskCollection.from_list expects Task or dict entries")
        return cls(tasks)

    @classmethod
    def from_json_file(cls, path: Union[str, Path], *, limit: Optional[int] = None) -> "TaskCollection":
        """Load tasks from a JSON file.

        This helper understands the example file format used in `examples/data.json` where the
        top-level object has a `data` list and optional `metadata`.

        Args:
            path: Path to the JSON file.
            limit: Optional limit to the number of tasks to load.
        """
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        # Support both the wrapped `{ "data": [...] }` format and a plain list
        items = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload

        if limit is not None:
            items = items[:limit]

        # Convert each item to a Task using the same heuristics as from_list
        return cls.from_list(items)
