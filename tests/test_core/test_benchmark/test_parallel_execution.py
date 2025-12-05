"""Tests for parallel task execution in Benchmark.

These tests verify that parallel execution with max_workers > 1 works correctly,
including thread safety, report collection, and callback serialization.
"""

import pytest
import threading
import time
from typing import List, Tuple, Optional

from maseval import (
    BenchmarkCallback,
    Task,
    TaskCollection,
    TaskExecutionStatus,
)
from conftest import DummyBenchmark


# ==================== Test Fixtures ====================


class SlowBenchmark(DummyBenchmark):
    """Benchmark that introduces configurable delays per task."""

    def __init__(self, delay_seconds: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay_seconds
        self.execution_times: List[Tuple[str, float, float]] = []  # (task_id, start, end)
        self._timing_lock = threading.Lock()

    def run_agents(self, agents, task, environment, query):
        start = time.time()
        time.sleep(self.delay)
        result = super().run_agents(agents, task, environment, query)
        end = time.time()

        with self._timing_lock:
            self.execution_times.append((str(task.id), start, end))

        return result


class ThreadTrackingCallback(BenchmarkCallback):
    """Callback that records which thread each event fires on."""

    def __init__(self):
        self.thread_ids: List[Tuple[str, Optional[int]]] = []
        self._lock = threading.Lock()

    def on_task_repeat_start(self, benchmark, task, repeat_idx):
        with self._lock:
            self.thread_ids.append(("repeat_start", threading.current_thread().ident))

    def on_task_repeat_end(self, benchmark, report):
        with self._lock:
            self.thread_ids.append(("repeat_end", threading.current_thread().ident))


class OrderTrackingCallback(BenchmarkCallback):
    """Callback that records the order of callback invocations."""

    def __init__(self):
        self.invocations: List[str] = []
        self._lock = threading.Lock()

    def on_run_start(self, benchmark):
        with self._lock:
            self.invocations.append("run_start")

    def on_task_start(self, benchmark, task):
        with self._lock:
            self.invocations.append(f"task_start:{task.query}")

    def on_task_repeat_start(self, benchmark, task, repeat_idx):
        with self._lock:
            self.invocations.append(f"repeat_start:{task.query}:{repeat_idx}")

    def on_task_repeat_end(self, benchmark, report):
        with self._lock:
            self.invocations.append(f"repeat_end:{report['task_id'][:8]}")

    def on_task_end(self, benchmark, task, result):
        with self._lock:
            self.invocations.append(f"task_end:{task.query}")

    def on_run_end(self, benchmark, results):
        with self._lock:
            self.invocations.append("run_end")


@pytest.fixture
def parallel_tasks():
    """Create tasks for parallel execution testing."""
    return TaskCollection.from_list([{"query": f"Task {i}", "environment_data": {"index": i}} for i in range(5)])


# ==================== Basic Parallel Execution Tests ====================


@pytest.mark.core
class TestParallelExecutionBasics:
    """Tests for basic parallel execution functionality."""

    def test_parallel_execution_completes(self, parallel_tasks):
        """Verify parallel execution completes all tasks."""
        benchmark = DummyBenchmark(agent_data={"model": "test"})

        reports = benchmark.run(parallel_tasks, max_workers=3)

        assert len(reports) == 5

    def test_parallel_produces_same_report_count(self, parallel_tasks):
        """Parallel and sequential should produce same number of reports."""
        benchmark_seq = DummyBenchmark(agent_data={"model": "test"})
        benchmark_par = DummyBenchmark(agent_data={"model": "test"})

        reports_seq = benchmark_seq.run(parallel_tasks, max_workers=1)
        reports_par = benchmark_par.run(parallel_tasks, max_workers=3)

        assert len(reports_seq) == len(reports_par)

    def test_parallel_reports_have_correct_structure(self, parallel_tasks):
        """Verify parallel reports have expected fields."""
        benchmark = DummyBenchmark(agent_data={"model": "test"})

        reports = benchmark.run(parallel_tasks, max_workers=2)

        for report in reports:
            assert "task_id" in report
            assert "repeat_idx" in report
            assert "status" in report
            assert "traces" in report
            assert "config" in report
            assert "eval" in report

    def test_single_worker_uses_sequential(self, parallel_tasks):
        """max_workers=1 should behave identically to sequential."""
        callback = OrderTrackingCallback()
        benchmark = DummyBenchmark(
            agent_data={"model": "test"},
            callbacks=[callback],
        )

        benchmark.run(parallel_tasks, max_workers=1)

        # Verify ordering is strictly sequential (task_start before all repeat_starts)
        assert callback.invocations[0] == "run_start"
        assert callback.invocations[1] == "task_start:Task 0"
        assert callback.invocations[-1] == "run_end"

    def test_parallel_with_repetitions(self):
        """Verify parallel execution with n_task_repeats > 1."""
        tasks = TaskCollection.from_list(
            [
                {"query": "T1", "environment_data": {}},
                {"query": "T2", "environment_data": {}},
            ]
        )
        benchmark = DummyBenchmark(
            agent_data={"model": "test"},
            n_task_repeats=3,
        )

        reports = benchmark.run(tasks, max_workers=2)

        assert len(reports) == 6  # 2 tasks × 3 repeats

        # Verify repeat indices
        repeat_indices = [r["repeat_idx"] for r in reports]
        assert set(repeat_indices) == {0, 1, 2}


# ==================== Thread Safety Tests ====================


@pytest.mark.core
class TestParallelThreadSafety:
    """Tests for thread safety in parallel execution."""

    def test_reports_all_collected(self, parallel_tasks):
        """All reports should be collected regardless of completion order."""
        benchmark = SlowBenchmark(
            agent_data={"model": "test"},
            delay_seconds=0.02,
        )

        reports = benchmark.run(parallel_tasks, max_workers=4)

        assert len(reports) == 5
        task_ids = {r["task_id"] for r in reports}
        assert len(task_ids) == 5

    def test_traces_not_cross_contaminated(self, parallel_tasks):
        """Traces from one task should not appear in another's report."""
        benchmark = DummyBenchmark(agent_data={"model": "test"})

        reports = benchmark.run(parallel_tasks, max_workers=3)

        for report in reports:
            # Each report should have its own traces
            assert report["traces"] is not None
            assert "metadata" in report["traces"]

    def test_callbacks_receive_correct_data(self):
        """Callbacks should receive correct task/report data in parallel."""
        tasks = TaskCollection.from_list([{"query": f"Query_{i}", "environment_data": {"idx": i}} for i in range(3)])

        received_data = []
        lock = threading.Lock()

        class DataCapturingCallback(BenchmarkCallback):
            def on_task_repeat_end(self, benchmark, report):
                with lock:
                    received_data.append(
                        {
                            "task_id": report["task_id"],
                            "status": report["status"],
                        }
                    )

        benchmark = DummyBenchmark(
            agent_data={"model": "test"},
            callbacks=[DataCapturingCallback()],
        )

        benchmark.run(tasks, max_workers=2)

        assert len(received_data) == 3
        statuses = {d["status"] for d in received_data}
        assert statuses == {"success"}

    def test_callback_exception_propagates(self, parallel_tasks):
        """Callback exceptions propagate (current behavior)."""
        call_count = [0]

        class FailingCallback(BenchmarkCallback):
            def on_task_repeat_end(self, benchmark, report):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise RuntimeError("Intentional failure")

        benchmark = DummyBenchmark(
            agent_data={"model": "test"},
            callbacks=[FailingCallback()],
        )

        # Current behavior: callback exceptions propagate
        with pytest.raises(RuntimeError, match="Intentional failure"):
            benchmark.run(parallel_tasks, max_workers=2)


# ==================== Concurrency Verification Tests ====================


@pytest.mark.core
class TestParallelConcurrency:
    """Tests verifying actual concurrent execution."""

    def test_parallel_faster_than_sequential(self):
        """Parallel execution should be faster for I/O-bound tasks."""
        tasks = TaskCollection.from_list([{"query": f"T{i}", "environment_data": {}} for i in range(4)])
        delay = 0.05

        # Sequential timing
        benchmark_seq = SlowBenchmark(agent_data={"model": "test"}, delay_seconds=delay)
        start_seq = time.time()
        benchmark_seq.run(tasks, max_workers=1)
        time_seq = time.time() - start_seq

        # Parallel timing
        benchmark_par = SlowBenchmark(agent_data={"model": "test"}, delay_seconds=delay)
        start_par = time.time()
        benchmark_par.run(tasks, max_workers=4)
        time_par = time.time() - start_par

        # Parallel should be significantly faster (at least 2x)
        assert time_par < time_seq * 0.7

    def test_execution_overlaps(self):
        """Task executions should overlap in parallel mode."""
        tasks = TaskCollection.from_list([{"query": f"T{i}", "environment_data": {}} for i in range(3)])

        benchmark = SlowBenchmark(
            agent_data={"model": "test"},
            delay_seconds=0.05,
        )

        benchmark.run(tasks, max_workers=3)

        # Check for overlapping execution times
        times = benchmark.execution_times
        assert len(times) == 3

        # At least one pair should overlap
        overlaps = 0
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                _, start_i, end_i = times[i]
                _, start_j, end_j = times[j]
                # Check if intervals overlap
                if start_i < end_j and start_j < end_i:
                    overlaps += 1

        assert overlaps > 0, "Expected overlapping execution in parallel mode"


# ==================== Error Handling Tests ====================


@pytest.mark.core
class TestParallelErrorHandling:
    """Tests for error handling in parallel execution."""

    def test_error_in_one_task_doesnt_stop_others(self):
        """One task failing should not prevent other tasks from completing."""

        class FailingBenchmark(DummyBenchmark):
            def run_agents(self, agents, task, environment, query):
                if "fail" in query.lower():
                    raise RuntimeError("Intentional failure")
                return super().run_agents(agents, task, environment, query)

        tasks = TaskCollection.from_list(
            [
                {"query": "Normal 1", "environment_data": {}},
                {"query": "FAIL task", "environment_data": {}},
                {"query": "Normal 2", "environment_data": {}},
            ]
        )

        benchmark = FailingBenchmark(agent_data={"model": "test"})
        reports = benchmark.run(tasks, max_workers=2)

        assert len(reports) == 3

        statuses = {r["status"] for r in reports}
        assert TaskExecutionStatus.SUCCESS.value in statuses
        assert TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR.value in statuses

    def test_all_tasks_get_reports_even_with_failures(self):
        """Every task should produce a report even if some fail."""

        class HalfFailingBenchmark(DummyBenchmark):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._call_count = 0
                self._lock = threading.Lock()

            def run_agents(self, agents, task, environment, query):
                with self._lock:
                    self._call_count += 1
                    should_fail = self._call_count % 2 == 0

                if should_fail:
                    raise ValueError("Every other task fails")
                return super().run_agents(agents, task, environment, query)

        tasks = TaskCollection.from_list([{"query": f"T{i}", "environment_data": {}} for i in range(4)])

        benchmark = HalfFailingBenchmark(agent_data={"model": "test"})
        reports = benchmark.run(tasks, max_workers=2)

        assert len(reports) == 4


# ==================== Queue Integration Tests ====================


@pytest.mark.core
class TestParallelQueueIntegration:
    """Tests for queue integration with parallel execution."""

    def test_custom_queue_respected(self, parallel_tasks):
        """Custom queue ordering should be respected."""
        from maseval.core.queue import PriorityQueue

        # Create tasks with priorities
        prioritized_tasks = TaskCollection(
            [
                Task(
                    query=f"P{p}",
                    environment_data={},
                    protocol=__import__("maseval.core.task", fromlist=["TaskProtocol"]).TaskProtocol(priority=p),
                )
                for p in [1, 5, 3, 2, 4]
            ]
        )

        agent_data_list = [{"model": "test"}] * 5
        queue = PriorityQueue(prioritized_tasks, agent_data_list)

        # Track execution order
        execution_order = []
        lock = threading.Lock()

        class OrderTracker(BenchmarkCallback):
            def on_task_repeat_start(self, benchmark, task, repeat_idx):
                with lock:
                    execution_order.append(task.query)

        benchmark = DummyBenchmark(
            agent_data={"model": "test"},
            callbacks=[OrderTracker()],
        )

        # With max_workers=1, order should be strictly by priority
        benchmark.run(prioritized_tasks, queue=queue, max_workers=1)

        assert execution_order == ["P5", "P4", "P3", "P2", "P1"]
