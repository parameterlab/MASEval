# Test Plan: Parallel Task Execution Engine

This document outlines the testing strategy for the parallel execution implementation. It covers new tests to add, existing tests to adapt, and tests that can be removed.

---

## 1. New Tests to Add

### 1.1 ComponentRegistry Tests (`tests/test_core/test_registry.py`)

**Thread Safety Tests:**

- `test_registry_thread_isolation` - Verify that registrations in one thread don't appear in another thread
- `test_registry_concurrent_registration` - Multiple threads registering components simultaneously without data races
- `test_registry_concurrent_collect_traces` - Multiple threads calling `collect_traces()` simultaneously
- `test_registry_concurrent_collect_configs` - Multiple threads calling `collect_configs()` simultaneously
- `test_registry_clear_only_affects_current_thread` - Calling `clear()` in one thread doesn't affect other threads

**Basic Functionality Tests:**

- `test_registry_register_traceable_component` - Component registered for tracing
- `test_registry_register_configurable_component` - Component also registered in config registry
- `test_registry_duplicate_key_idempotent` - Same component, same key is idempotent
- `test_registry_duplicate_component_different_key_raises` - Same component, different key raises ValueError
- `test_registry_collect_traces_structure` - Verify trace output structure
- `test_registry_collect_configs_structure` - Verify config output structure
- `test_registry_benchmark_config_included` - benchmark_config passed to constructor appears in configs

### 1.2 TaskContext Tests (`tests/test_core/test_context.py`)

**Timeout Behavior Tests:**

- `test_context_no_timeout` - Context without deadline never expires
- `test_context_with_timeout_not_expired` - Context before deadline shows remaining time
- `test_context_with_timeout_expired` - Context after deadline shows is_expired=True
- `test_context_check_timeout_raises_on_expiry` - `check_timeout()` raises TaskTimeoutError when expired
- `test_context_check_timeout_with_partial_traces` - TaskTimeoutError includes partial traces
- `test_context_elapsed_increases` - `elapsed` property increases over time
- `test_context_remaining_decreases` - `remaining` property decreases over time

### 1.3 TaskQueue Tests (`tests/test_core/test_queue.py`)

**SequentialQueue Tests:**

- `test_sequential_queue_order_preserved` - Tasks yielded in original order
- `test_sequential_queue_iteration_complete` - All tasks yielded exactly once
- `test_sequential_queue_empty_collection` - Empty collection yields nothing
- `test_sequential_queue_single_task` - Single task handled correctly

**PriorityQueue Tests:**

- `test_priority_queue_high_priority_first` - Higher priority tasks come first
- `test_priority_queue_stable_sort` - Equal priority maintains original order
- `test_priority_queue_default_priority_zero` - Tasks without explicit priority treated as 0
- `test_priority_queue_negative_priority` - Negative priorities handled correctly

**AdaptiveQueue Tests:**

- `test_adaptive_queue_on_task_complete_updates_state` - Completed tasks moved to completed list
- `test_adaptive_queue_stop_terminates_iteration` - Calling `stop()` ends iteration early
- `test_adaptive_queue_should_continue_false_after_stop` - `should_continue()` returns False after stop

### 1.4 TaskProtocol Tests (`tests/test_core/test_task_protocol.py`)

- `test_task_protocol_defaults` - Default values: timeout=None, action=SKIP, retries=0, priority=0, tags={}
- `test_task_has_protocol_field` - Task dataclass has protocol field
- `test_task_protocol_custom_values` - Custom protocol values preserved
- `test_timeout_action_enum_values` - TimeoutAction has SKIP, RETRY, RAISE

### 1.5 Parallel Execution Tests (`tests/test_core/test_benchmark/test_parallel_execution.py`)

**Basic Parallel Execution:**

- `test_parallel_execution_basic` - `max_workers>1` runs tasks in parallel
- `test_parallel_execution_same_results_as_sequential` - Parallel produces same reports as sequential
- `test_parallel_execution_max_workers_respected` - No more than max_workers concurrent threads
- `test_parallel_execution_single_worker_uses_sequential` - `max_workers=1` uses `_run_sequential`

**Thread Safety - Report Collection:**

- `test_parallel_reports_thread_safe` - Reports from parallel tasks all collected correctly
- `test_parallel_report_count_matches_task_count` - Number of reports equals tasks × repeats
- `test_parallel_report_order_independent` - Report content correct regardless of completion order

**Thread Safety - Callbacks:**

- `test_parallel_callbacks_serialized` - Callbacks invoked with lock (no concurrent callback execution)
- `test_parallel_callback_data_integrity` - Callback receives correct task/report data
- `test_parallel_callbacks_all_events_fire` - All lifecycle callbacks fire for each task
- `test_parallel_callback_exception_isolated` - Exception in one callback doesn't affect other tasks

**Thread Safety - Registry:**

- `test_parallel_registry_isolation` - Each task gets isolated registry state
- `test_parallel_traces_not_cross_contaminated` - Traces from task A don't appear in task B's report
- `test_parallel_configs_not_cross_contaminated` - Configs from task A don't appear in task B's report

**Race Condition Tests:**

- `test_parallel_concurrent_setup` - Multiple tasks calling setup methods simultaneously
- `test_parallel_concurrent_evaluation` - Multiple tasks being evaluated simultaneously
- `test_parallel_slow_fast_task_ordering` - Slow task in worker doesn't block fast task reports
- `test_parallel_error_in_one_task_doesnt_affect_others` - One task failing doesn't corrupt other tasks

### 1.6 Timeout Handling Tests (`tests/test_core/test_benchmark/test_timeout_handling.py`)

- `test_timeout_task_marked_as_timeout_status` - Timed out task has `TASK_TIMEOUT` status
- `test_timeout_partial_traces_collected` - Traces collected up to timeout point included in report
- `test_timeout_action_skip_continues_to_next` - SKIP action moves to next task
- `test_timeout_action_retry_retries_task` - RETRY action re-executes (up to max_retries)
- `test_timeout_action_raise_propagates` - RAISE action raises TaskTimeoutError
- `test_timeout_cooperative_checkpoint` - Tasks checking `context.check_timeout()` respect timeout

### 1.7 Queue Integration Tests (`tests/test_core/test_benchmark/test_queue_integration.py`)

- `test_run_with_custom_queue` - `benchmark.run(tasks, queue=custom_queue)` uses provided queue
- `test_run_default_queue_is_sequential` - No queue specified uses SequentialQueue
- `test_priority_queue_integration` - PriorityQueue orders execution correctly in real benchmark
- `test_queue_on_task_complete_called` - Queue's `on_task_complete` called after each task
- `test_queue_should_continue_checked` - Queue's `should_continue` checked after each task

### 1.8 TaskTimeoutError Tests (`tests/test_core/test_exceptions.py` - extend existing)

- `test_task_timeout_error_attributes` - Has elapsed, timeout, partial_traces attributes
- `test_task_timeout_error_message` - Message includes timeout and elapsed time
- `test_task_timeout_error_is_maseval_error` - Inherits from MASEvalError

---

## 2. Existing Tests to Adapt

### 2.1 Tests Already Adapted (completed)

- `tests/test_core/test_benchmark/test_automatic_registration.py`
  - Changed `benchmark._trace_registry` → `benchmark._registry._trace_registry`
  - Changed `benchmark._component_id_map` → `benchmark._registry._component_id_map`
- `tests/test_core/test_benchmark/test_benchmark_lifecycle.py`
  - Changed `benchmark._trace_registry` → `benchmark._registry._trace_registry`
  - Changed `benchmark._config_registry` → `benchmark._registry._config_registry`

### 2.2 Tests That May Need Adaptation

**Callback Tests (`test_callback_orchestration.py`):**

- Review `test_callback_errors_dont_break_execution` - Ensure behavior consistent with parallel mode
- Consider adding parallel variant of each callback order test

**Lifecycle Tests (`test_benchmark_lifecycle.py`):**

- `test_benchmark_lifecycle_hooks_order` - Verify order still guaranteed in sequential mode
- Add note/variant about callback order in parallel mode (order within task preserved, between tasks not)

**Exception Tests (`test_exceptions.py`):**

- Extend classification tests to include `TaskTimeoutError` → `TASK_TIMEOUT` mapping

**Config Collection Tests (`test_config_collection.py`):**

- Verify config collection works correctly in parallel mode
- `test_config_different_per_repetition` - May need thread-awareness verification

---

## 3. Tests That Can Be Removed

### 3.1 No Tests to Remove

The implementation maintains backward compatibility (`max_workers=1` default), so all existing tests remain valid. No tests are obsoleted by this change.

### 3.2 Tests That Could Be Consolidated (Optional Cleanup)

- Some registry-related tests in `test_automatic_registration.py` and `test_benchmark_lifecycle.py` overlap in testing registry clearing. Consider consolidating into a single registry test file.

---

## 4. Test Categories and Markers

### New Pytest Markers to Consider

```python
# conftest.py additions
pytest.mark.parallel  # Tests specific to parallel execution
pytest.mark.thread_safety  # Tests for race conditions and thread safety
pytest.mark.timeout  # Tests for timeout handling
pytest.mark.queue  # Tests for task queue abstraction
```

### Marker Usage

```python
@pytest.mark.core
@pytest.mark.parallel
def test_parallel_execution_basic():
    ...

@pytest.mark.core
@pytest.mark.thread_safety
def test_parallel_registry_isolation():
    ...
```

---

## 5. Test Infrastructure Needs

### 5.1 New Test Fixtures

```python
# conftest.py additions

@pytest.fixture
def slow_benchmark():
    """Benchmark that takes configurable time per task (for parallel testing)."""
    class SlowBenchmark(DummyBenchmark):
        def __init__(self, delay_seconds=0.1, **kwargs):
            super().__init__(**kwargs)
            self.delay = delay_seconds

        def run_agents(self, agents, task, environment, query):
            import time
            time.sleep(self.delay)
            return super().run_agents(agents, task, environment, query)

    return SlowBenchmark

@pytest.fixture
def thread_tracking_callback():
    """Callback that records which thread each event fires on."""
    import threading

    class ThreadTracker(BenchmarkCallback):
        def __init__(self):
            self.thread_ids = []

        def on_task_repeat_start(self, benchmark, task, repeat_idx):
            self.thread_ids.append(threading.current_thread().ident)

    return ThreadTracker
```

### 5.2 Helper Functions

```python
def run_parallel_and_sequential(benchmark, tasks):
    """Run same benchmark both ways and compare reports."""
    import copy

    seq_benchmark = copy.deepcopy(benchmark)
    par_benchmark = copy.deepcopy(benchmark)

    seq_reports = seq_benchmark.run(tasks, max_workers=1)
    par_reports = par_benchmark.run(tasks, max_workers=4)

    return seq_reports, par_reports

def verify_no_cross_contamination(reports):
    """Check that traces in each report only contain that task's data."""
    for report in reports:
        task_id = report['task_id']
        for key, trace in report['traces'].get('agents', {}).items():
            # Verify trace belongs to this task
            assert task_id in str(trace) or 'task_id' not in trace
```

---

## 6. Priority Order for Implementation

### High Priority (Core Functionality)

1. `test_parallel_execution.py` - Basic parallel execution verification
2. `test_registry.py` - Thread isolation is critical for correctness
3. `test_timeout_handling.py` - Timeout is a key new feature

### Medium Priority (Integration)

4. `test_queue.py` - Queue abstraction tests
5. `test_queue_integration.py` - Queue + Benchmark integration
6. `test_context.py` - TaskContext functionality

### Lower Priority (Edge Cases)

7. `test_task_protocol.py` - Simple dataclass tests
8. Extended race condition tests
9. Performance/stress tests

---

## 7. Notes for Test Implementation

### Thread Safety Testing Patterns

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor

def test_concurrent_operation():
    """Pattern for testing concurrent operations."""
    results = []
    errors = []
    barrier = threading.Barrier(4)  # Synchronize thread start

    def worker(worker_id):
        try:
            barrier.wait()  # All threads start together
            # Perform operation
            result = do_something()
            results.append((worker_id, result))
        except Exception as e:
            errors.append((worker_id, e))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        for f in futures:
            f.result()  # Wait for completion

    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 4
```

### Timing Considerations

- Use `time.sleep()` sparingly in tests
- Consider mocking time for deterministic timeout tests
- Use threading barriers for synchronization points
- Allow tolerance in timing assertions (e.g., ±10ms)

### Isolation Verification

```python
def test_registry_isolation():
    """Verify thread-local storage works correctly."""
    registry = ComponentRegistry()
    results = {}

    def worker(worker_id):
        # Each thread should see empty registry initially
        assert len(registry._trace_registry) == 0

        # Register unique component
        registry.register("test", f"comp_{worker_id}", MockComponent())

        # Only our component should be visible
        assert len(registry._trace_registry) == 1
        assert f"test:comp_{worker_id}" in registry._trace_registry

        results[worker_id] = list(registry._trace_registry.keys())

    # Run in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        for f in futures:
            f.result()

    # Verify isolation
    for worker_id, keys in results.items():
        assert keys == [f"test:comp_{worker_id}"]
```
