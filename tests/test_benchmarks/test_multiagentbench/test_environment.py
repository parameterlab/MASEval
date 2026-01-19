"""Tests for MultiAgentBench environment."""

import pytest
from typing import Any, Dict
from unittest.mock import patch

from maseval.benchmark.multiagentbench.environment import (
    MultiAgentBenchEnvironment,
    INFRASTRUCTURE_DOMAINS,
)
from maseval import EnvironmentError


class TestInfrastructureDomains:
    """Tests for infrastructure domain constants."""

    def test_infrastructure_domains_contains_expected(self):
        """INFRASTRUCTURE_DOMAINS should contain expected domains."""
        assert "database" in INFRASTRUCTURE_DOMAINS
        assert "minecraft" in INFRASTRUCTURE_DOMAINS

    def test_infrastructure_domains_excludes_simple(self):
        """INFRASTRUCTURE_DOMAINS should not include simple domains."""
        assert "research" not in INFRASTRUCTURE_DOMAINS
        assert "bargaining" not in INFRASTRUCTURE_DOMAINS


class TestMultiAgentBenchEnvironment:
    """Tests for MultiAgentBenchEnvironment class."""

    def test_init_with_research_task(self, sample_research_task_data: Dict[str, Any]):
        """Environment should initialize for research domain."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        assert env.domain == "research"
        assert env.state is not None

    def test_init_with_bargaining_task(self, sample_bargaining_task_data: Dict[str, Any]):
        """Environment should initialize for bargaining domain."""
        env = MultiAgentBenchEnvironment(task_data=sample_bargaining_task_data)

        assert env.domain == "bargaining"

    def test_setup_state_extracts_domain(self, sample_research_task_data: Dict[str, Any]):
        """setup_state should extract domain from task data."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)
        state = env.state

        assert state["domain"] == "research"

    def test_setup_state_extracts_max_iterations(self, sample_research_task_data: Dict[str, Any]):
        """setup_state should extract max_iterations."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)
        state = env.state

        assert state["max_iterations"] == 10

    def test_is_done_initially_false(self, sample_research_task_data: Dict[str, Any]):
        """is_done should return False initially."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        # Without MARBLE env, always returns False
        assert env.is_done() is False

    def test_is_task_completed_initially_false(self, sample_research_task_data: Dict[str, Any]):
        """is_task_completed should return False initially."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        # Without MARBLE env, always returns False
        assert env.is_task_completed() is False

    def test_get_marble_state_empty_without_marble(self, sample_research_task_data: Dict[str, Any]):
        """get_marble_state should return empty dict without MARBLE."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        assert env.get_marble_state() == {}

    def test_get_tool_descriptions_empty_without_marble(self, sample_research_task_data: Dict[str, Any]):
        """get_tool_descriptions should return empty dict without MARBLE."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        assert env.get_tool_descriptions() == {}

    def test_create_tools_empty_without_marble(self, sample_research_task_data: Dict[str, Any]):
        """create_tools should return empty dict without MARBLE."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)
        tools = env.create_tools()

        assert tools == {}

    def test_gather_traces_includes_domain(self, sample_research_task_data: Dict[str, Any]):
        """gather_traces should include domain information."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)
        traces = env.gather_traces()

        assert traces["domain"] == "research"
        assert "tool_invocations" in traces

    def test_gather_config_includes_domain(self, sample_research_task_data: Dict[str, Any]):
        """gather_config should include domain information."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)
        config = env.gather_config()

        assert config["domain"] == "research"
        assert "tool_descriptions" in config


class TestInfrastructureCheck:
    """Tests for infrastructure checking."""

    def test_database_without_docker_raises(self):
        """Environment should raise for database without Docker."""
        task_data = {
            "scenario": "database",
            "environment": {"type": "DB"},
            "task": {"content": "Query database"},
            "agents": [{"agent_id": "agent1"}],
        }

        with patch("shutil.which", return_value=None):
            with pytest.raises(EnvironmentError, match="requires external infrastructure"):
                MultiAgentBenchEnvironment(task_data=task_data)

    def test_database_with_docker_succeeds(self):
        """Environment should succeed for database with Docker."""
        task_data = {
            "scenario": "database",
            "environment": {"type": "DB"},
            "task": {"content": "Query database"},
            "agents": [{"agent_id": "agent1"}],
        }

        with patch("shutil.which", return_value="/usr/bin/docker"):
            # Should not raise, but MARBLE env creation may still fail
            try:
                env = MultiAgentBenchEnvironment(task_data=task_data)
                assert env.domain == "database"
            except ImportError:
                # Expected if MARBLE not available
                pass

    def test_minecraft_always_raises(self):
        """Environment should raise for minecraft (not supported)."""
        task_data = {
            "scenario": "minecraft",
            "environment": {"type": "Minecraft"},
            "task": {"content": "Build something"},
            "agents": [{"agent_id": "agent1"}],
        }

        with pytest.raises(EnvironmentError, match="requires external infrastructure"):
            MultiAgentBenchEnvironment(task_data=task_data)


class TestApplyAction:
    """Tests for apply_action method."""

    def test_apply_action_without_marble_raises(self, sample_research_task_data: Dict[str, Any]):
        """apply_action should raise without MARBLE environment."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        with pytest.raises(EnvironmentError, match="not available"):
            env.apply_action("agent1", "some_action", {"arg": "value"})
