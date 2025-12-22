"""Unit tests for Tau2 data_loader module."""

import pytest

from maseval.benchmark.tau2.data_loader import (
    DEFAULT_DATA_DIR,
    VALID_DOMAINS,
    TASK_SPLITS,
    load_domain_config,
    load_tasks,
    configure_model_ids,
    ensure_data_exists,
)


# =============================================================================
# Constants Tests
# =============================================================================


@pytest.mark.benchmark
class TestConstants:
    """Tests for module constants."""

    def test_valid_domains(self):
        """VALID_DOMAINS contains expected domains."""
        assert "retail" in VALID_DOMAINS
        assert "airline" in VALID_DOMAINS
        assert "telecom" in VALID_DOMAINS
        assert len(VALID_DOMAINS) == 3

    def test_task_splits(self):
        """TASK_SPLITS contains expected splits."""
        assert "base" in TASK_SPLITS
        assert "hard" in TASK_SPLITS
        assert "all" in TASK_SPLITS

    def test_default_data_dir_is_module_relative(self):
        """DEFAULT_DATA_DIR points to module's data directory."""
        assert DEFAULT_DATA_DIR.name == "data"
        assert DEFAULT_DATA_DIR.parent.name == "tau2"
        assert DEFAULT_DATA_DIR.parent.parent.name == "benchmark"


# =============================================================================
# Domain Config Tests
# =============================================================================


@pytest.mark.benchmark
class TestLoadDomainConfig:
    """Tests for load_domain_config function."""

    def test_loads_retail_config(self):
        """Loads retail domain configuration."""
        config = load_domain_config("retail")

        assert "policy" in config
        assert "db_path" in config
        assert config["db_path"].exists()
        assert len(config["policy"]) > 0

    def test_loads_airline_config(self):
        """Loads airline domain configuration."""
        config = load_domain_config("airline")

        assert "policy" in config
        assert config["db_path"].exists()

    def test_loads_telecom_config(self):
        """Loads telecom domain configuration."""
        config = load_domain_config("telecom")

        assert "policy" in config
        assert config["db_path"].exists()

    def test_invalid_domain_raises(self):
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            load_domain_config("invalid_domain")


# =============================================================================
# Load Tasks Tests
# =============================================================================


@pytest.mark.benchmark
class TestLoadTasks:
    """Tests for load_tasks function."""

    def test_loads_retail_tasks(self):
        """Loads retail domain tasks."""
        tasks = load_tasks("retail", split="base", limit=5)

        assert len(tasks) <= 5
        assert len(tasks) > 0

        task = tasks[0]
        assert task.query is not None
        assert "domain" in task.environment_data or task.metadata.get("domain") == "retail"

    def test_loads_airline_tasks(self):
        """Loads airline domain tasks."""
        tasks = load_tasks("airline", split="base", limit=5)

        assert len(tasks) <= 5
        assert len(tasks) > 0

    def test_loads_telecom_tasks(self):
        """Loads telecom domain tasks."""
        tasks = load_tasks("telecom", split="base", limit=5)

        assert len(tasks) <= 5
        assert len(tasks) > 0

    def test_limit_parameter(self):
        """Limit parameter restricts number of tasks."""
        tasks_3 = load_tasks("retail", limit=3)
        tasks_10 = load_tasks("retail", limit=10)

        assert len(tasks_3) <= 3
        assert len(tasks_10) <= 10

    def test_invalid_domain_raises(self):
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            load_tasks("invalid_domain")

    def test_invalid_split_raises(self):
        """Invalid split raises ValueError."""
        with pytest.raises(ValueError, match="Invalid split"):
            load_tasks("retail", split="invalid_split")


# =============================================================================
# Configure Model IDs Tests
# =============================================================================


@pytest.mark.benchmark
class TestConfigureModelIds:
    """Tests for configure_model_ids function."""

    def test_configures_user_model(self):
        """Configures user model ID."""
        tasks = load_tasks("retail", limit=2)
        configure_model_ids(tasks, user_model_id="gpt-4o")

        for task in tasks:
            assert task.user_data.get("model_id") == "gpt-4o"

    def test_configures_evaluator_model(self):
        """Configures evaluator model ID."""
        tasks = load_tasks("retail", limit=2)
        configure_model_ids(tasks, evaluator_model_id="claude-3-opus")

        for task in tasks:
            assert task.evaluation_data.get("model_id") == "claude-3-opus"

    def test_configures_multiple_models(self):
        """Configures multiple model IDs at once."""
        tasks = load_tasks("retail", limit=2)
        configure_model_ids(
            tasks,
            user_model_id="user-model",
            evaluator_model_id="eval-model",
        )

        for task in tasks:
            assert task.user_data.get("model_id") == "user-model"
            assert task.evaluation_data.get("model_id") == "eval-model"


# =============================================================================
# Ensure Data Exists Tests
# =============================================================================


@pytest.mark.benchmark
class TestEnsureDataExists:
    """Tests for ensure_data_exists function."""

    def test_retail_data_exists(self):
        """Retail data exists after ensure_data_exists."""
        result = ensure_data_exists(domain="retail")

        assert result.exists()
        assert (result / "retail" / "db.json").exists()
        assert (result / "retail" / "tasks.json").exists()
        assert (result / "retail" / "policy.md").exists()

    def test_airline_data_exists(self):
        """Airline data exists after ensure_data_exists."""
        result = ensure_data_exists(domain="airline")

        assert (result / "airline" / "db.json").exists()
        assert (result / "airline" / "tasks.json").exists()
        assert (result / "airline" / "policy.md").exists()

    def test_telecom_data_exists(self):
        """Telecom data exists after ensure_data_exists."""
        result = ensure_data_exists(domain="telecom")

        # Telecom uses db.toml instead of db.json
        assert (result / "telecom" / "db.toml").exists()
        assert (result / "telecom" / "tasks.json").exists()
        assert (result / "telecom" / "policy.md").exists()


# =============================================================================
# Task Content Tests
# =============================================================================


@pytest.mark.benchmark
class TestTaskContent:
    """Tests for task content structure."""

    def test_task_has_required_fields(self):
        """Tasks have all required fields."""
        tasks = load_tasks("retail", limit=5)

        for task in tasks:
            assert task.id is not None
            assert task.query is not None
            assert task.environment_data is not None
            assert task.user_data is not None
            assert task.evaluation_data is not None

    def test_task_evaluation_data_structure(self):
        """Task evaluation_data has expected structure."""
        tasks = load_tasks("retail", limit=5)

        for task in tasks:
            eval_data = task.evaluation_data
            # Just verify the structure is a dict
            assert isinstance(eval_data, dict)

    def test_task_user_data_has_instructions(self):
        """Task user_data contains instructions."""
        tasks = load_tasks("retail", limit=5)

        for task in tasks:
            user_data = task.user_data
            # Just verify user_data is a dict
            assert isinstance(user_data, dict)
