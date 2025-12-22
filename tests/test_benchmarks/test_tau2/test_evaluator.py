"""Unit tests for Tau2Evaluator."""

import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict

from maseval import Task
from maseval.benchmark.tau2.evaluator import Tau2Evaluator, RewardType


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
        "actions": [
            {"name": "check_order", "arguments": {"order_id": "123"}}
        ],
        "communicate_info": ["refund processed"],
        "env_assertions": [
            {"func_name": "check_status", "arguments": {"id": "1"}, "assert_value": True}
        ]
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
        "agents": {
            "agent1": {
                "messages": [{"role": "assistant", "content": "Hello"}]
            }
        },
        "tools": {
            "tool1": {
                "invocations": [
                    {"inputs": {"a": 1}, "outputs": "res", "status": "success"}
                ]
            }
        },
        "environment": {"final_db_hash": "hash123"},
        "termination_reason": "agent_stop"
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
    traces = {
        "environment": {"final_db_hash": "hash_gold"}
    }
    
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
    traces = {
        "environment": {"final_db_hash": "hash_actual"}
    }
    
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
    traces = {
        "tool_calls": [
            {"name": "check_order", "arguments": {"order_id": "123"}}
        ]
    }
    
    result = evaluator._evaluate_actions(traces)
    
    assert result["all_matched"] is True
    assert result["reward"] == 1.0


@pytest.mark.benchmark
def test_evaluate_actions_mismatch(evaluator):
    """Test action evaluation with mismatching tool calls."""
    traces = {
        "tool_calls": [
            {"name": "check_order", "arguments": {"order_id": "999"}} # Wrong ID
        ]
    }
    
    result = evaluator._evaluate_actions(traces)
    
    assert result["all_matched"] is False
    assert result["reward"] == 0.0


@pytest.mark.benchmark
def test_evaluate_actions_missing(evaluator):
    """Test action evaluation with missing tool calls."""
    traces = {
        "tool_calls": []
    }
    
    result = evaluator._evaluate_actions(traces)
    
    assert result["all_matched"] is False
    assert result["reward"] == 0.0


# =============================================================================
# Communication Evaluation Tests
# =============================================================================

@pytest.mark.benchmark
def test_evaluate_communication_success(evaluator):
    """Test communication evaluation finding required info."""
    traces = {
        "messages": [
            {"content": "Your refund processed successfully."}
        ]
    }
    
    result = evaluator._evaluate_communication(traces)
    
    assert result["all_found"] is True
    assert result["reward"] == 1.0


@pytest.mark.benchmark
def test_evaluate_communication_failure(evaluator):
    """Test communication evaluation failing to find info."""
    traces = {
        "messages": [
            {"content": "I cannot help you."}
        ]
    }
    
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
    
    assert result["reward"] == 0.0 # Multiplicative
    assert result["passed"] is False


@pytest.mark.benchmark
def test_premature_termination(evaluator):
    """Test evaluation aborts on error termination."""
    traces = {"termination_reason": "too_many_errors"}
    
    result = evaluator(traces)
    
    assert result["reward"] == 0.0
    assert result["passed"] is False
    assert "prematurely" in result["note"]
