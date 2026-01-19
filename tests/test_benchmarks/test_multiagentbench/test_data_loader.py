"""Tests for MultiAgentBench data loading functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch
import tempfile
import json

from maseval import Task
from maseval.benchmark.multiagentbench.data_loader import (
    load_tasks,
    configure_model_ids,
    get_domain_info,
    VALID_DOMAINS,
    _parse_task_entry,
    _resolve_data_dir,
)


class TestValidDomains:
    """Tests for domain validation."""

    def test_valid_domains_contains_expected(self):
        """VALID_DOMAINS should contain all expected domains."""
        expected = {"coding", "database", "minecraft", "research", "bargaining", "web", "worldsimulation"}
        assert expected == VALID_DOMAINS

    def test_valid_domains_is_frozen(self):
        """VALID_DOMAINS should be immutable."""
        assert isinstance(VALID_DOMAINS, frozenset)


class TestGetDomainInfo:
    """Tests for get_domain_info function."""

    def test_research_domain_info(self):
        """get_domain_info should return correct info for research."""
        info = get_domain_info("research")
        assert info["requires_infrastructure"] is False
        assert info["coordination_mode"] == "cooperative"
        assert "description" in info

    def test_database_requires_infrastructure(self):
        """Database domain should require infrastructure."""
        info = get_domain_info("database")
        assert info["requires_infrastructure"] is True

    def test_minecraft_requires_infrastructure(self):
        """Minecraft domain should require infrastructure."""
        info = get_domain_info("minecraft")
        assert info["requires_infrastructure"] is True

    def test_invalid_domain_raises(self):
        """get_domain_info should raise for invalid domain."""
        with pytest.raises(ValueError, match="Invalid domain"):
            get_domain_info("invalid_domain")

    def test_case_insensitive(self):
        """get_domain_info should be case-insensitive."""
        info_lower = get_domain_info("research")
        info_upper = get_domain_info("RESEARCH")
        assert info_lower == info_upper


class TestParseTaskEntry:
    """Tests for _parse_task_entry function."""

    def test_parse_minimal_entry(self):
        """_parse_task_entry should parse a minimal valid entry."""
        entry = {
            "scenario": "research",
            "task_id": 1,
            "task": {"content": "Do research", "output_format": "5Q format"},
            "agents": [{"agent_id": "agent1", "profile": "Researcher"}],
            "relationships": [["agent1", "agent1", "self"]],
        }
        task = _parse_task_entry(entry, "research", 0)

        assert isinstance(task, Task)
        assert task.id == "research_1"
        assert task.query == "Do research"
        assert task.environment_data["scenario"] == "research"
        assert len(task.environment_data["agents"]) == 1

    def test_parse_entry_missing_required_field(self):
        """_parse_task_entry should raise for missing required fields."""
        entry = {
            "scenario": "research",
            "task_id": 1,
            # Missing "task", "agents", "relationships"
        }
        with pytest.raises(ValueError, match="missing required fields"):
            _parse_task_entry(entry, "research", 0)

    def test_parse_entry_missing_agent_id(self):
        """_parse_task_entry should raise if agent missing agent_id."""
        entry = {
            "scenario": "research",
            "task_id": 1,
            "task": {"content": "Do research"},
            "agents": [{"profile": "Researcher"}],  # Missing agent_id
            "relationships": [],
        }
        with pytest.raises(ValueError, match="missing 'agent_id'"):
            _parse_task_entry(entry, "research", 0)

    def test_parse_entry_empty_query(self):
        """_parse_task_entry should raise for empty query."""
        entry = {
            "scenario": "research",
            "task_id": 1,
            "task": {"content": "", "output_format": "5Q format"},
            "agents": [{"agent_id": "agent1"}],
            "relationships": [],
        }
        with pytest.raises(ValueError, match="empty query"):
            _parse_task_entry(entry, "research", 0)

    def test_parse_entry_with_string_task(self):
        """_parse_task_entry should handle task as string."""
        entry = {
            "scenario": "research",
            "task_id": 1,
            "task": "Do research task",
            "agents": [{"agent_id": "agent1"}],
            "relationships": [],
        }
        task = _parse_task_entry(entry, "research", 0)
        assert task.query == "Do research task"

    def test_parse_entry_preserves_metadata(self):
        """_parse_task_entry should preserve metadata correctly."""
        entry = {
            "scenario": "bargaining",
            "task_id": 42,
            "task": {"content": "Negotiate"},
            "agents": [{"agent_id": "buyer"}],
            "relationships": [],
            "coordinate_mode": "star",
            "environment": {"max_iterations": 20},
        }
        task = _parse_task_entry(entry, "bargaining", 0)

        assert task.metadata["domain"] == "bargaining"
        assert task.metadata["task_id"] == 42
        assert task.environment_data["coordinate_mode"] == "star"
        assert task.environment_data["max_iterations"] == 20


class TestLoadTasks:
    """Tests for load_tasks function."""

    def test_load_tasks_invalid_domain(self):
        """load_tasks should raise for invalid domain."""
        with pytest.raises(ValueError, match="Invalid domain"):
            load_tasks("invalid_domain")

    def test_load_tasks_missing_data_dir(self):
        """load_tasks should raise if data directory not found."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(FileNotFoundError, match="does not exist"):
                load_tasks("research", data_dir=Path("/nonexistent/path"))

    def test_load_tasks_with_mock_data(self):
        """load_tasks should load tasks from JSONL file."""
        # Create temporary JSONL file
        with tempfile.TemporaryDirectory() as tmpdir:
            research_dir = Path(tmpdir) / "research"
            research_dir.mkdir()
            jsonl_path = research_dir / "research_main.jsonl"

            # Write sample task
            task_data = {
                "scenario": "research",
                "task_id": 1,
                "task": {"content": "Research task", "output_format": "5Q"},
                "agents": [{"agent_id": "agent1", "profile": "Researcher"}],
                "relationships": [],
            }
            with jsonl_path.open("w") as f:
                f.write(json.dumps(task_data) + "\n")

            tasks = load_tasks("research", data_dir=Path(tmpdir))

            assert len(tasks) == 1
            assert tasks[0].query == "Research task"

    def test_load_tasks_with_limit(self):
        """load_tasks should respect limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            research_dir = Path(tmpdir) / "research"
            research_dir.mkdir()
            jsonl_path = research_dir / "research_main.jsonl"

            # Write multiple tasks
            with jsonl_path.open("w") as f:
                for i in range(5):
                    task_data = {
                        "scenario": "research",
                        "task_id": i + 1,
                        "task": {"content": f"Research task {i + 1}"},
                        "agents": [{"agent_id": f"agent{i + 1}"}],
                        "relationships": [],
                    }
                    f.write(json.dumps(task_data) + "\n")

            tasks = load_tasks("research", data_dir=Path(tmpdir), limit=2)

            assert len(tasks) == 2

    def test_load_tasks_case_insensitive_domain(self):
        """load_tasks should handle domain case-insensitively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            research_dir = Path(tmpdir) / "research"
            research_dir.mkdir()
            jsonl_path = research_dir / "research_main.jsonl"

            task_data = {
                "scenario": "research",
                "task_id": 1,
                "task": {"content": "Test"},
                "agents": [{"agent_id": "agent1"}],
                "relationships": [],
            }
            with jsonl_path.open("w") as f:
                f.write(json.dumps(task_data) + "\n")

            tasks_lower = load_tasks("research", data_dir=Path(tmpdir))
            tasks_upper = load_tasks("RESEARCH", data_dir=Path(tmpdir))

            assert len(tasks_lower) == len(tasks_upper) == 1


class TestConfigureModelIds:
    """Tests for configure_model_ids function."""

    def test_configure_model_ids_sets_llm(self):
        """configure_model_ids should set llm in environment_data."""
        task = Task(
            id="test_1",
            query="Test query",
            environment_data={"scenario": "research"},
            evaluation_data={},
            metadata={},
        )
        tasks = [task]

        configure_model_ids(tasks, agent_model_id="gpt-4o")

        assert tasks[0].environment_data["llm"] == "gpt-4o"

    def test_configure_model_ids_sets_evaluator_model(self):
        """configure_model_ids should set evaluator model_id."""
        task = Task(
            id="test_1",
            query="Test query",
            environment_data={},
            evaluation_data={},
            metadata={},
        )
        tasks = [task]

        configure_model_ids(
            tasks,
            agent_model_id="gpt-4o",
            evaluator_model_id="gpt-4o-mini",
        )

        assert tasks[0].evaluation_data["model_id"] == "gpt-4o-mini"

    def test_configure_model_ids_defaults_evaluator_to_agent(self):
        """configure_model_ids should default evaluator model to agent model."""
        task = Task(
            id="test_1",
            query="Test query",
            environment_data={},
            evaluation_data={},
            metadata={},
        )
        tasks = [task]

        configure_model_ids(tasks, agent_model_id="gpt-4o")

        assert tasks[0].evaluation_data["model_id"] == "gpt-4o"

    def test_configure_model_ids_returns_tasks(self):
        """configure_model_ids should return the input tasks."""
        task = Task(
            id="test_1",
            query="Test query",
            environment_data={},
            evaluation_data={},
            metadata={},
        )
        tasks = [task]

        result = configure_model_ids(tasks, agent_model_id="gpt-4o")

        assert result is tasks


class TestResolveDataDir:
    """Tests for _resolve_data_dir function."""

    def test_resolve_explicit_path(self):
        """_resolve_data_dir should use explicit path if provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _resolve_data_dir(Path(tmpdir))
            assert result == Path(tmpdir)

    def test_resolve_nonexistent_explicit_path(self):
        """_resolve_data_dir should raise for nonexistent explicit path."""
        with pytest.raises(FileNotFoundError):
            _resolve_data_dir(Path("/nonexistent/path"))

    def test_resolve_from_env_var(self):
        """_resolve_data_dir should use MARBLE_DATA_DIR env var."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"MARBLE_DATA_DIR": tmpdir}):
                result = _resolve_data_dir()
                assert result == Path(tmpdir)
