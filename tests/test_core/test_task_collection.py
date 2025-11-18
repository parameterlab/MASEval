"""Test TaskCollection functionality.

These tests verify that TaskCollection behaves like a sequence and correctly
loads tasks from various formats.
"""

import pytest
from maseval import Task, TaskCollection
from pathlib import Path
import json
import tempfile


@pytest.mark.core
class TestTaskCollection:
    """Tests for TaskCollection interface and factories."""

    def test_task_collection_from_list(self):
        """Test creating TaskCollection from a list of dicts."""
        data = [
            {"query": "Q1", "environment_data": {"e": 1}},
            {"query": "Q2", "environment_data": {"e": 2}},
        ]

        collection = TaskCollection.from_list(data)

        assert len(collection) == 2
        assert collection[0].query == "Q1"
        assert collection[1].query == "Q2"

    def test_task_collection_from_json_file(self):
        """Test loading TaskCollection from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test JSON file
            data = {
                "data": [
                    {"query": "Task 1", "environment_data": {}},
                    {"query": "Task 2", "environment_data": {}},
                ]
            }

            file_path = Path(tmpdir) / "tasks.json"
            with open(file_path, "w") as f:
                json.dump(data, f)

            # Load from file
            collection = TaskCollection.from_json_file(file_path)

            assert len(collection) == 2
            assert collection[0].query == "Task 1"
            assert collection[1].query == "Task 2"

    def test_task_collection_sequence_interface(self):
        """Test that TaskCollection implements Sequence interface."""
        collection = TaskCollection.from_list(
            [
                {"query": "Q1"},
                {"query": "Q2"},
                {"query": "Q3"},
            ]
        )

        # Test len
        assert len(collection) == 3

        # Test iteration
        queries = [task.query for task in collection]
        assert queries == ["Q1", "Q2", "Q3"]

        # Test indexing
        assert collection[0].query == "Q1"
        assert collection[-1].query == "Q3"

    def test_task_collection_slicing(self):
        """Test that TaskCollection supports slicing."""
        collection = TaskCollection.from_list([{"query": f"Q{i}"} for i in range(10)])

        # Test slice
        subset = collection[2:5]
        assert isinstance(subset, TaskCollection)
        assert len(subset) == 3
        assert subset[0].query == "Q2"
        assert subset[2].query == "Q4"

        # Test slice from start
        start = collection[:3]
        assert len(start) == 3

        # Test slice to end
        end = collection[7:]
        assert len(end) == 3

    def test_task_collection_iteration(self):
        """Test iterating over TaskCollection."""
        data = [{"query": f"Q{i}"} for i in range(5)]
        collection = TaskCollection.from_list(data)

        queries = []
        for task in collection:
            queries.append(task.query)

        assert queries == ["Q0", "Q1", "Q2", "Q3", "Q4"]

    def test_task_dict_conversion(self):
        """Test that dict items are converted to Task objects."""
        collection = TaskCollection.from_list(
            [
                {
                    "query": "Test",
                    "environment_data": {"key": "value"},
                    "evaluation_data": {"expected": "result"},
                    "metadata": {"difficulty": "easy"},
                }
            ]
        )

        task = collection[0]
        assert isinstance(task, Task)
        assert task.query == "Test"
        assert task.environment_data == {"key": "value"}
        assert task.evaluation_data == {"expected": "result"}
        assert task.metadata == {"difficulty": "easy"}

    def test_task_field_mapping(self):
        """Test that alternative field names are mapped correctly."""
        # Test question -> query mapping
        collection = TaskCollection.from_list([{"question": "What is 2+2?", "short_answer": "4"}])

        task = collection[0]
        assert task.query == "What is 2+2?"
        assert task.evaluation_data == {"short_answer": "4"}

    def test_task_collection_append(self):
        """Test appending tasks to collection."""
        collection = TaskCollection()
        assert len(collection) == 0

        task = Task(query="Test")
        collection.append(task)

        assert len(collection) == 1
        assert collection[0].query == "Test"

    def test_task_collection_extend(self):
        """Test extending collection with multiple tasks."""
        collection = TaskCollection()

        new_tasks = [
            Task(query="Q1"),
            Task(query="Q2"),
            Task(query="Q3"),
        ]

        collection.extend(new_tasks)

        assert len(collection) == 3
        assert collection[2].query == "Q3"

    def test_task_collection_to_list(self):
        """Test converting TaskCollection to list."""
        data = [{"query": f"Q{i}"} for i in range(3)]
        collection = TaskCollection.from_list(data)

        task_list = collection.to_list()

        assert isinstance(task_list, list)
        assert len(task_list) == 3
        assert all(isinstance(t, Task) for t in task_list)

    def test_task_collection_from_json_with_limit(self):
        """Test loading with a limit on number of tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"data": [{"query": f"Task {i}"} for i in range(10)]}

            file_path = Path(tmpdir) / "tasks.json"
            with open(file_path, "w") as f:
                json.dump(data, f)

            # Load only first 5
            collection = TaskCollection.from_json_file(file_path, limit=5)

            assert len(collection) == 5
            assert collection[4].query == "Task 4"

    def test_task_collection_repr(self):
        """Test string representation of TaskCollection."""
        collection = TaskCollection.from_list([{"query": "Q1"}, {"query": "Q2"}])

        repr_str = repr(collection)
        assert "TaskCollection" in repr_str
        assert "2" in repr_str  # Should mention number of tasks
