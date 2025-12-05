"""Tests for TaskQueue implementations.

These tests verify that SequentialQueue, PriorityQueue, and AdaptiveQueue
correctly order and iterate over tasks.
"""

import pytest
from typing import Any, Dict, List

from maseval import Task, TaskCollection
from maseval.core.task import TaskProtocol
from maseval.core.queue import SequentialQueue, PriorityQueue, AdaptiveQueue


# ==================== Fixtures ====================


@pytest.fixture
def task_collection_with_priorities() -> TaskCollection:
    """Create tasks with different priorities."""
    tasks = []
    for i, priority in enumerate([0, 5, 2, 8, 1]):
        task = Task(
            query=f"Query {i}",
            environment_data={"index": i},
            protocol=TaskProtocol(priority=priority),
        )
        tasks.append(task)
    return TaskCollection(tasks)


@pytest.fixture
def agent_data_list() -> List[Dict[str, Any]]:
    """Agent data list matching 5 tasks."""
    return [{"id": i} for i in range(5)]


@pytest.fixture
def simple_task_collection() -> TaskCollection:
    """Simple task collection for basic tests."""
    return TaskCollection.from_list(
        [
            {"query": "Q1", "environment_data": {}},
            {"query": "Q2", "environment_data": {}},
            {"query": "Q3", "environment_data": {}},
        ]
    )


@pytest.fixture
def simple_agent_data() -> List[Dict[str, Any]]:
    """Agent data matching simple_task_collection."""
    return [{"model": "test"}] * 3


# ==================== SequentialQueue Tests ====================


@pytest.mark.core
class TestSequentialQueue:
    """Tests for SequentialQueue ordering."""

    def test_order_preserved(self, simple_task_collection, simple_agent_data):
        """Tasks should be yielded in original order."""
        queue = SequentialQueue(simple_task_collection, simple_agent_data)

        queries = [task.query for task, _ in queue]

        assert queries == ["Q1", "Q2", "Q3"]

    def test_all_tasks_yielded(self, simple_task_collection, simple_agent_data):
        """All tasks should be yielded exactly once."""
        queue = SequentialQueue(simple_task_collection, simple_agent_data)

        count = sum(1 for _ in queue)

        assert count == 3

    def test_empty_collection(self):
        """Empty collection should yield nothing."""
        queue = SequentialQueue(TaskCollection([]), [])

        items = list(queue)

        assert items == []

    def test_single_task(self):
        """Single task should be handled correctly."""
        tasks = TaskCollection.from_list([{"query": "Only one"}])
        queue = SequentialQueue(tasks, [{"model": "test"}])

        items = list(queue)

        assert len(items) == 1
        assert items[0][0].query == "Only one"

    def test_agent_data_paired_correctly(self, simple_task_collection):
        """Agent data should be paired with correct task."""
        agent_data = [{"id": 1}, {"id": 2}, {"id": 3}]
        queue = SequentialQueue(simple_task_collection, agent_data)

        pairs = list(queue)

        assert pairs[0][1]["id"] == 1
        assert pairs[1][1]["id"] == 2
        assert pairs[2][1]["id"] == 3


# ==================== PriorityQueue Tests ====================


@pytest.mark.core
class TestPriorityQueue:
    """Tests for PriorityQueue priority ordering."""

    def test_high_priority_first(self, task_collection_with_priorities, agent_data_list):
        """Higher priority tasks should come first."""
        queue = PriorityQueue(task_collection_with_priorities, agent_data_list)

        priorities = [task.protocol.priority for task, _ in queue]

        assert priorities == [8, 5, 2, 1, 0]

    def test_stable_sort_for_equal_priorities(self):
        """Tasks with equal priority should maintain original order."""
        tasks = TaskCollection(
            [
                Task(query="First", environment_data={}, protocol=TaskProtocol(priority=5)),
                Task(query="Second", environment_data={}, protocol=TaskProtocol(priority=5)),
                Task(query="Third", environment_data={}, protocol=TaskProtocol(priority=5)),
            ]
        )
        agent_data = [{}, {}, {}]
        queue = PriorityQueue(tasks, agent_data)

        queries = [task.query for task, _ in queue]

        # Python's sort is stable, so original order should be preserved
        assert queries == ["First", "Second", "Third"]

    def test_default_priority_zero(self, simple_task_collection, simple_agent_data):
        """Tasks without explicit priority should have priority 0."""
        queue = PriorityQueue(simple_task_collection, simple_agent_data)

        for task, _ in queue:
            assert task.protocol.priority == 0

    def test_negative_priority(self):
        """Negative priorities should be handled correctly."""
        tasks = TaskCollection(
            [
                Task(query="Low", environment_data={}, protocol=TaskProtocol(priority=-5)),
                Task(query="Normal", environment_data={}, protocol=TaskProtocol(priority=0)),
                Task(query="High", environment_data={}, protocol=TaskProtocol(priority=5)),
            ]
        )
        queue = PriorityQueue(tasks, [{}, {}, {}])

        queries = [task.query for task, _ in queue]

        assert queries == ["High", "Normal", "Low"]

    def test_agent_data_follows_priority(self, task_collection_with_priorities, agent_data_list):
        """Agent data should follow task after priority sort."""
        queue = PriorityQueue(task_collection_with_priorities, agent_data_list)

        pairs = list(queue)

        # Task with priority 8 was at index 3
        assert pairs[0][1]["id"] == 3
        # Task with priority 5 was at index 1
        assert pairs[1][1]["id"] == 1


# ==================== AdaptiveQueue Tests ====================


@pytest.mark.core
class TestAdaptiveQueue:
    """Tests for AdaptiveQueue adaptive behavior."""

    def test_basic_iteration_with_completion(self, simple_task_collection, simple_agent_data):
        """AdaptiveQueue should yield all tasks when on_task_complete is called."""
        queue = AdaptiveQueue(simple_task_collection, simple_agent_data)

        count = 0
        for task, agent_data in queue:
            count += 1
            # Must call on_task_complete to progress to next task
            queue.on_task_complete(task, {"status": "success"})

        assert count == 3

    def test_on_task_complete_moves_to_completed(self, simple_task_collection, simple_agent_data):
        """on_task_complete should move task to completed list."""
        queue = AdaptiveQueue(simple_task_collection, simple_agent_data)
        task, _ = next(iter(queue))

        assert len(queue._completed) == 0

        queue.on_task_complete(task, {"status": "success"})

        assert len(queue._completed) == 1
        assert queue._completed[0][0].id == task.id

    def test_stop_terminates_iteration(self, simple_task_collection, simple_agent_data):
        """Calling stop() should end iteration early."""
        queue = AdaptiveQueue(simple_task_collection, simple_agent_data)

        items = []
        for task, agent_data in queue:
            items.append(task)
            queue.stop()  # Stop immediately after first yield

        assert len(items) == 1

    def test_should_continue_false_after_stop(self, simple_task_collection, simple_agent_data):
        """should_continue() should return False after stop()."""
        queue = AdaptiveQueue(simple_task_collection, simple_agent_data)

        assert queue.should_continue() is True

        queue.stop()

        assert queue.should_continue() is False

    def test_should_continue_false_when_empty(self):
        """should_continue() should return False when no pending tasks."""
        queue = AdaptiveQueue(TaskCollection([]), [])

        assert queue.should_continue() is False

    def test_pending_decreases_after_completion(self, simple_task_collection, simple_agent_data):
        """Pending list should shrink as tasks complete."""
        queue = AdaptiveQueue(simple_task_collection, simple_agent_data)

        assert len(queue._pending) == 3

        task, _ = next(iter(queue))
        queue.on_task_complete(task, {"status": "success"})

        assert len(queue._pending) == 2
        assert len(queue._completed) == 1


# ==================== Queue Integration Tests ====================


@pytest.mark.core
class TestQueueCallbacks:
    """Tests for queue callback mechanisms."""

    def test_on_task_complete_called(self, simple_task_collection, simple_agent_data):
        """on_task_complete should be callable without error."""
        queue = SequentialQueue(simple_task_collection, simple_agent_data)

        for task, _ in queue:
            # SequentialQueue's on_task_complete is a no-op, but should not raise
            queue.on_task_complete(task, {"status": "success"})

    def test_should_continue_always_true_for_sequential(self, simple_task_collection, simple_agent_data):
        """SequentialQueue should always return True for should_continue."""
        queue = SequentialQueue(simple_task_collection, simple_agent_data)

        for task, _ in queue:
            assert queue.should_continue() is True
