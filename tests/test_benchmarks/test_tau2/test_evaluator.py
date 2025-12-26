"""Unit tests for Tau2Evaluator."""

import pytest
from unittest.mock import MagicMock, patch

from maseval import Task
from maseval.benchmark.tau2.evaluator import Tau2Evaluator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_environment():
    env = MagicMock()
    env.domain = "retail"
    # Default hash values
    env.get_db_hash.return_value = "hash123"
    env.toolkit.has_tool.return_value = True
    env.toolkit.use_tool.return_value = True
    return env


@pytest.fixture
def sample_task():
    task = MagicMock(spec=Task)
    task.environment_data = {"domain": "retail"}
    task.evaluation_data = {
        "reward_basis": ["DB", "ACTION", "COMMUNICATE"],
        "actions": [{"name": "check_order", "arguments": {"order_id": "123"}}],
        "communicate_info": ["refund processed"],
        "env_assertions": [{"func_name": "check_status", "arguments": {"id": "1"}, "assert_value": True}],
    }
    return task


@pytest.fixture
def evaluator(sample_task, mock_environment):
    return Tau2Evaluator(sample_task, mock_environment)


# =============================================================================
# Trace Filtering Tests
# =============================================================================


@pytest.mark.benchmark
def test_filter_traces(evaluator):
    """Test extraction of relevant traces."""
    raw_traces = {
        "agents": {"agent1": {"messages": [{"role": "assistant", "content": "Hello"}]}},
        "tools": {"tool1": {"invocations": [{"inputs": {"a": 1}, "outputs": "res", "status": "success"}]}},
        "environment": {"final_db_hash": "hash123"},
        "termination_reason": "agent_stop",
    }

    filtered = evaluator.filter_traces(raw_traces)

    assert len(filtered["messages"]) == 1
    assert filtered["messages"][0]["content"] == "Hello"

    assert len(filtered["tool_calls"]) == 1
    assert filtered["tool_calls"][0]["name"] == "tool1"
    assert filtered["tool_calls"][0]["arguments"] == {"a": 1}

    assert filtered["environment"]["final_db_hash"] == "hash123"
    assert filtered["termination_reason"] == "agent_stop"


# =============================================================================
# Environment Evaluation Tests
# =============================================================================


@pytest.mark.benchmark
def test_evaluate_environment_success(evaluator, mock_environment):
    """Test environment evaluation with matching hashes."""
    traces = {"environment": {"final_db_hash": "hash_gold"}}

    # Mock gold environment creation
    with patch("maseval.benchmark.tau2.evaluator.get_environment_constructor") as mock_get_const:
        mock_gold_env = MagicMock()
        mock_gold_env.get_db_hash.return_value = "hash_gold"
        mock_get_const.return_value.return_value = mock_gold_env

        result = evaluator._evaluate_environment(traces)

        assert result["db_match"] is True
        assert result["db_reward"] == 1.0
        assert result["reward"] == 1.0  # Combined reward (including assertion which passes by default fixture)


@pytest.mark.benchmark
def test_evaluate_environment_failure(evaluator, mock_environment):
    """Test environment evaluation with mismatching hashes."""
    traces = {"environment": {"final_db_hash": "hash_actual"}}

    with patch("maseval.benchmark.tau2.evaluator.get_environment_constructor") as mock_get_const:
        mock_gold_env = MagicMock()
        mock_gold_env.get_db_hash.return_value = "hash_expected"
        mock_get_const.return_value.return_value = mock_gold_env

        result = evaluator._evaluate_environment(traces)

        assert result["db_match"] is False
        assert result["db_reward"] == 0.0


@pytest.mark.benchmark
def test_evaluate_env_assertion(evaluator, mock_environment):
    """Test environment assertion logic."""
    # Ensure tool exists and returns True
    mock_environment.toolkit.has_tool.return_value = True
    mock_environment.toolkit.use_tool.return_value = True

    assertion = {"func_name": "check", "arguments": {}, "assert_value": True}

    assert evaluator._run_env_assertion(mock_environment, assertion) is True

    # Test mismatch
    mock_environment.toolkit.use_tool.return_value = False
    assert evaluator._run_env_assertion(mock_environment, assertion) is False


# =============================================================================
# Action Evaluation Tests
# =============================================================================


@pytest.mark.benchmark
def test_evaluate_actions_match(evaluator):
    """Test action evaluation with matching tool calls."""
    traces = {"tool_calls": [{"name": "check_order", "arguments": {"order_id": "123"}}]}

    result = evaluator._evaluate_actions(traces)

    assert result["all_matched"] is True
    assert result["reward"] == 1.0


@pytest.mark.benchmark
def test_evaluate_actions_mismatch(evaluator):
    """Test action evaluation with mismatching tool calls."""
    traces = {
        "tool_calls": [
            {"name": "check_order", "arguments": {"order_id": "999"}}  # Wrong ID
        ]
    }

    result = evaluator._evaluate_actions(traces)

    assert result["all_matched"] is False
    assert result["reward"] == 0.0


@pytest.mark.benchmark
def test_evaluate_actions_missing(evaluator):
    """Test action evaluation with missing tool calls."""
    traces = {"tool_calls": []}

    result = evaluator._evaluate_actions(traces)

    assert result["all_matched"] is False
    assert result["reward"] == 0.0


# =============================================================================
# Communication Evaluation Tests
# =============================================================================


@pytest.mark.benchmark
def test_evaluate_communication_success(evaluator):
    """Test communication evaluation finding required info."""
    traces = {"messages": [{"content": "Your refund processed successfully."}]}

    result = evaluator._evaluate_communication(traces)

    assert result["all_found"] is True
    assert result["reward"] == 1.0


@pytest.mark.benchmark
def test_evaluate_communication_failure(evaluator):
    """Test communication evaluation failing to find info."""
    traces = {"messages": [{"content": "I cannot help you."}]}

    result = evaluator._evaluate_communication(traces)

    assert result["all_found"] is False
    assert result["reward"] == 0.0


# =============================================================================
# Score Aggregation Tests
# =============================================================================


@pytest.mark.benchmark
def test_score_aggregation_all_pass(evaluator):
    """Test overall score when all components pass."""
    # Mock individual methods to return success
    evaluator._evaluate_environment = MagicMock(return_value={"reward": 1.0, "breakdown": {"db": 1.0}})
    evaluator._evaluate_actions = MagicMock(return_value={"reward": 1.0, "breakdown": {"action": 1.0}})
    evaluator._evaluate_communication = MagicMock(return_value={"reward": 1.0, "breakdown": {"communicate": 1.0}})

    result = evaluator({"termination_reason": "agent_stop"})

    assert result["reward"] == 1.0
    assert result["passed"] is True
    assert result["reward_breakdown"] == {"db": 1.0, "action": 1.0, "communicate": 1.0}


@pytest.mark.benchmark
def test_score_aggregation_mixed(evaluator):
    """Test overall score when some components fail."""
    # Environment fails, others pass
    evaluator._evaluate_environment = MagicMock(return_value={"reward": 0.0, "breakdown": {"db": 0.0}})
    evaluator._evaluate_actions = MagicMock(return_value={"reward": 1.0, "breakdown": {"action": 1.0}})
    evaluator._evaluate_communication = MagicMock(return_value={"reward": 1.0, "breakdown": {"communicate": 1.0}})

    result = evaluator({"termination_reason": "agent_stop"})

    assert result["reward"] == 0.0  # Multiplicative
    assert result["passed"] is False


@pytest.mark.benchmark
def test_premature_termination(evaluator):
    """Test evaluation aborts on error termination."""
    traces = {"termination_reason": "too_many_errors"}

    result = evaluator(traces)

    assert result["reward"] == 0.0
    assert result["passed"] is False
    assert "prematurely" in result["note"]


@pytest.mark.benchmark
def test_max_steps_termination(evaluator):
    """Test evaluation aborts on max_steps termination."""
    traces = {"termination_reason": "max_steps"}

    result = evaluator(traces)

    assert result["reward"] == 0.0
    assert result["passed"] is False
    assert "prematurely" in result["note"]


# =============================================================================
# Metrics Tests
# =============================================================================


@pytest.mark.benchmark
class TestComputeBenchmarkMetrics:
    """Tests for compute_benchmark_metrics function."""

    def test_empty_results(self):
        """Empty results returns zeros."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        result = compute_benchmark_metrics([])

        assert result["total_tasks"] == 0
        assert result["scored_tasks"] == 0
        assert result["successful_tasks"] == 0
        assert result["success_rate"] == 0.0
        assert result["mean_reward"] == 0.0

    def test_single_success(self):
        """Single successful result counted."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        results = [{"status": "completed", "eval": [{"reward": 1.0, "passed": True}]}]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 1
        assert metrics["scored_tasks"] == 1
        assert metrics["successful_tasks"] == 1
        assert metrics["success_rate"] == 1.0
        assert metrics["mean_reward"] == 1.0

    def test_single_failure(self):
        """Single failed result counted."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        results = [{"status": "completed", "eval": [{"reward": 0.0, "passed": False}]}]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 1
        assert metrics["scored_tasks"] == 1
        assert metrics["successful_tasks"] == 0
        assert metrics["success_rate"] == 0.0

    def test_mixed_results(self):
        """Mixed success/failure results aggregated."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        results = [
            {"status": "completed", "eval": [{"reward": 1.0, "passed": True}]},
            {"status": "completed", "eval": [{"reward": 0.0, "passed": False}]},
            {"status": "completed", "eval": [{"reward": 0.5, "passed": False}]},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 3
        assert metrics["scored_tasks"] == 3
        assert metrics["successful_tasks"] == 1
        assert metrics["success_rate"] == pytest.approx(1 / 3)
        assert metrics["mean_reward"] == pytest.approx(0.5)

    def test_excludes_infrastructure_errors(self):
        """Infrastructure errors excluded from scoring."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        results = [
            {"status": "completed", "eval": [{"reward": 1.0, "passed": True}]},
            {"status": "environment_error", "eval": None},
            {"status": "user_error", "eval": None},
            {"status": "setup_failed", "eval": None},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 4
        assert metrics["scored_tasks"] == 1  # Only completed
        assert metrics["successful_tasks"] == 1
        assert metrics["success_rate"] == 1.0

    def test_status_counts(self):
        """Status counts tracked correctly."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        results = [
            {"status": "completed", "eval": [{"reward": 1.0, "passed": True}]},
            {"status": "completed", "eval": [{"reward": 0.0, "passed": False}]},
            {"status": "environment_error", "eval": None},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["status_counts"]["completed"] == 2
        assert metrics["status_counts"]["environment_error"] == 1


@pytest.mark.benchmark
class TestComputePassAtK:
    """Tests for compute_pass_at_k function."""

    def test_all_pass(self):
        """All attempts pass gives 1.0 for all k."""
        from maseval.benchmark.tau2.evaluator import compute_pass_at_k

        results = [
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
        ]

        pass_k = compute_pass_at_k(results, k_values=[1, 2, 3])

        assert pass_k["pass@1"] == 1.0
        assert pass_k["pass@2"] == 1.0
        assert pass_k["pass@3"] == 1.0

    def test_all_fail(self):
        """All attempts fail gives 0.0 for all k."""
        from maseval.benchmark.tau2.evaluator import compute_pass_at_k

        results = [
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
        ]

        pass_k = compute_pass_at_k(results, k_values=[1, 2, 3])

        assert pass_k["pass@1"] == 0.0
        assert pass_k["pass@2"] == 0.0
        assert pass_k["pass@3"] == 0.0

    def test_mixed_results(self):
        """Mixed results with pass on second attempt."""
        from maseval.benchmark.tau2.evaluator import compute_pass_at_k

        results = [
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
        ]

        pass_k = compute_pass_at_k(results, k_values=[1, 2, 3])

        assert pass_k["pass@1"] == 0.0  # First attempt failed
        assert pass_k["pass@2"] == 1.0  # Second attempt passed
        assert pass_k["pass@3"] == 1.0  # At least one passed

    def test_insufficient_attempts(self):
        """Insufficient attempts for k returns 0.0."""
        from maseval.benchmark.tau2.evaluator import compute_pass_at_k

        results = [
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
        ]

        pass_k = compute_pass_at_k(results, k_values=[1, 2, 3])

        assert pass_k["pass@1"] == 1.0
        assert pass_k["pass@2"] == 0.0  # Not enough attempts
        assert pass_k["pass@3"] == 0.0

    def test_multiple_tasks(self):
        """Multiple tasks with different outcomes."""
        from maseval.benchmark.tau2.evaluator import compute_pass_at_k

        results = [
            # Task 1: passes on first try
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            # Task 2: fails all
            {"task_id": "task2", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task2", "status": "success", "eval": [{"passed": False}]},
        ]

        pass_k = compute_pass_at_k(results, k_values=[1, 2])

        assert pass_k["pass@1"] == 0.5  # 1/2 tasks pass@1
        assert pass_k["pass@2"] == 0.5  # 1/2 tasks pass@2
