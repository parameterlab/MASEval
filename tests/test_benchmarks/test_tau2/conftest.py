"""Shared fixtures for Tau 2 benchmark tests.

Fixture Hierarchy
-----------------
- tests/conftest.py: Generic fixtures (DummyModelAdapter, dummy_model, etc.)
- tests/test_benchmarks/test_tau2/conftest.py: Tau2-specific fixtures (this file)

Tau2 tests can use fixtures from both levels - pytest handles this automatically.
"""

import pytest
from tempfile import TemporaryDirectory
from pathlib import Path

from maseval import Task


# =============================================================================
# Session-Scoped Setup
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def ensure_tau2_data():
    """Download Tau2 domain data before running any benchmark tests.

    This fixture runs once per test session and ensures that the domain
    data files (db.json, tasks.json, policy.md) exist locally.
    """
    from maseval.benchmark.tau2.data_loader import ensure_data_exists

    for domain in ["retail", "airline", "telecom"]:
        ensure_data_exists(domain=domain)


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Domain Database Fixtures
# =============================================================================


@pytest.fixture
def retail_db():
    """Load the retail domain database."""
    from maseval.benchmark.tau2.data_loader import load_domain_config, DEFAULT_DATA_DIR
    from maseval.benchmark.tau2.domains.retail import RetailDB

    config = load_domain_config("retail", DEFAULT_DATA_DIR)
    return RetailDB.load(config["db_path"])


@pytest.fixture
def airline_db():
    """Load the airline domain database."""
    from maseval.benchmark.tau2.data_loader import load_domain_config, DEFAULT_DATA_DIR
    from maseval.benchmark.tau2.domains.airline import AirlineDB

    config = load_domain_config("airline", DEFAULT_DATA_DIR)
    return AirlineDB.load(config["db_path"])


@pytest.fixture
def telecom_db():
    """Load the telecom domain database."""
    from maseval.benchmark.tau2.data_loader import load_domain_config, DEFAULT_DATA_DIR
    from maseval.benchmark.tau2.domains.telecom import TelecomDB

    config = load_domain_config("telecom", DEFAULT_DATA_DIR)
    return TelecomDB.load(config["db_path"])


# =============================================================================
# Toolkit Fixtures
# =============================================================================


@pytest.fixture
def retail_toolkit(retail_db):
    """Create a retail toolkit with database."""
    from maseval.benchmark.tau2.domains.retail import RetailTools

    return RetailTools(retail_db)


@pytest.fixture
def airline_toolkit(airline_db):
    """Create an airline toolkit with database."""
    from maseval.benchmark.tau2.domains.airline import AirlineTools

    return AirlineTools(airline_db)


@pytest.fixture
def telecom_toolkit(telecom_db):
    """Create a telecom toolkit with database."""
    from maseval.benchmark.tau2.domains.telecom import TelecomTools

    return TelecomTools(telecom_db)


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def retail_environment():
    """Create a retail environment."""
    from maseval.benchmark.tau2 import Tau2Environment

    return Tau2Environment({"domain": "retail"})


@pytest.fixture
def airline_environment():
    """Create an airline environment."""
    from maseval.benchmark.tau2 import Tau2Environment

    return Tau2Environment({"domain": "airline"})


@pytest.fixture
def telecom_environment():
    """Create a telecom environment."""
    from maseval.benchmark.tau2 import Tau2Environment

    return Tau2Environment({"domain": "telecom"})


# =============================================================================
# Task Fixtures
# =============================================================================


@pytest.fixture
def sample_retail_task():
    """Sample retail domain task."""
    return Task(
        query="I want to cancel my order",
        environment_data={"domain": "retail"},
        user_data={
            "model_id": "test-model",
            "instructions": {"reason_for_call": "Cancel order #12345"},
        },
        evaluation_data={"model_id": "test-model"},
        metadata={"domain": "retail"},
    )


@pytest.fixture
def sample_airline_task():
    """Sample airline domain task."""
    return Task(
        query="I need to change my flight",
        environment_data={"domain": "airline"},
        user_data={
            "model_id": "test-model",
            "instructions": {"reason_for_call": "Change flight reservation"},
        },
        evaluation_data={"model_id": "test-model"},
        metadata={"domain": "airline"},
    )


@pytest.fixture
def sample_telecom_task():
    """Sample telecom domain task."""
    return Task(
        query="My internet is not working",
        environment_data={"domain": "telecom"},
        user_data={
            "model_id": "test-model",
            "instructions": {"reason_for_call": "Internet connectivity issue"},
        },
        evaluation_data={"model_id": "test-model"},
        metadata={"domain": "telecom"},
    )
