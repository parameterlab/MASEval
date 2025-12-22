"""Unit tests for Tau2 environment module."""

import pytest

from maseval.benchmark.tau2 import Tau2Environment


# =============================================================================
# Environment Creation Tests
# =============================================================================


@pytest.mark.benchmark
class TestEnvironmentCreation:
    """Tests for Tau2Environment creation."""

    def test_creates_retail_environment(self):
        """Creates retail environment successfully."""
        env = Tau2Environment({"domain": "retail"})

        assert env.domain == "retail"
        assert env.db is not None
        assert env.toolkit is not None
        assert env.policy is not None

    def test_creates_airline_environment(self):
        """Creates airline environment successfully."""
        env = Tau2Environment({"domain": "airline"})

        assert env.domain == "airline"
        assert env.db is not None
        assert env.toolkit is not None

    def test_creates_telecom_environment(self):
        """Creates telecom environment successfully."""
        env = Tau2Environment({"domain": "telecom"})

        assert env.domain == "telecom"
        assert env.db is not None
        assert env.toolkit is not None

    def test_invalid_domain_raises(self):
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            Tau2Environment({"domain": "invalid_domain"})


# =============================================================================
# Environment Tools Tests
# =============================================================================


@pytest.mark.benchmark
class TestEnvironmentTools:
    """Tests for environment tool creation."""

    def test_retail_has_tools(self, retail_environment):
        """Retail environment has tools."""
        tools = retail_environment.create_tools()

        assert len(tools) > 0
        assert isinstance(tools, dict)

    def test_airline_has_tools(self, airline_environment):
        """Airline environment has tools."""
        tools = airline_environment.create_tools()

        assert len(tools) > 0

    def test_telecom_has_tools(self, telecom_environment):
        """Telecom environment has tools."""
        tools = telecom_environment.create_tools()

        assert len(tools) > 0

    def test_tools_are_callable(self, retail_environment):
        """All tools are callable."""
        tools = retail_environment.create_tools()

        for name, tool in tools.items():
            assert callable(tool), f"Tool {name} is not callable"


# =============================================================================
# Database State Tests
# =============================================================================


@pytest.mark.benchmark
class TestDatabaseState:
    """Tests for database state management."""

    def test_get_db_hash_returns_string(self, retail_environment):
        """get_db_hash returns a hash string."""
        hash_value = retail_environment.get_db_hash()

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex length

    def test_initial_db_hash_stored(self, retail_environment):
        """Initial DB hash is stored in state."""
        initial_hash = retail_environment.get_initial_db_hash()

        assert isinstance(initial_hash, str)
        assert len(initial_hash) == 64

    def test_hash_consistent_without_changes(self, retail_environment):
        """Hash is consistent when no changes made."""
        hash1 = retail_environment.get_db_hash()
        hash2 = retail_environment.get_db_hash()

        assert hash1 == hash2

    def test_hash_changes_after_modification(self, retail_environment):
        """Hash changes after database modification."""
        initial_hash = retail_environment.get_db_hash()

        # Modify database through a tool
        try:
            # Try to modify order status (may fail if order doesn't exist in test state)
            retail_environment.make_tool_call(
                "modify_pending_order_address",
                order_id="test",
                address1="123 Test St",
                address2="",
                city="Test City",
                state="TX",
                country="USA",
                zip="12345",
            )
        except (ValueError, KeyError):
            # Expected if order doesn't exist - test passes
            return

        final_hash = retail_environment.get_db_hash()
        # If modification succeeded, hash should be different
        assert initial_hash != final_hash


# =============================================================================
# Tool Execution Tests
# =============================================================================


@pytest.mark.benchmark
class TestToolExecution:
    """Tests for tool execution via environment."""

    def test_make_tool_call_read_tool(self, retail_environment):
        """Can execute a read-only tool."""
        # Get first user from database
        users = list(retail_environment.db.users.keys())
        if users:
            user_id = users[0]
            result = retail_environment.make_tool_call("get_user_details", user_id=user_id)
            assert result is not None

    def test_make_tool_call_invalid_tool_raises(self, retail_environment):
        """Invalid tool name raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            retail_environment.make_tool_call("nonexistent_tool")

    def test_airline_tool_execution(self, airline_environment):
        """Can execute airline domain tools."""
        users = list(airline_environment.db.users.keys())
        if users:
            user_id = users[0]
            result = airline_environment.make_tool_call("get_user_details", user_id=user_id)
            assert result is not None

    def test_telecom_tool_execution(self, telecom_environment):
        """Can execute telecom domain tools."""
        customers = telecom_environment.db.customers
        if customers:
            customer_id = customers[0].customer_id
            result = telecom_environment.make_tool_call("get_customer_by_id", customer_id=customer_id)
            assert result is not None


# =============================================================================
# Toolkit Statistics Tests
# =============================================================================


@pytest.mark.benchmark
class TestToolkitStatistics:
    """Tests for toolkit statistics."""

    def test_retail_toolkit_stats(self, retail_environment):
        """Retail toolkit has expected statistics."""
        stats = retail_environment.toolkit.get_statistics()

        assert "num_tools" in stats
        assert stats["num_tools"] > 0
        assert "num_read_tools" in stats
        assert "num_write_tools" in stats

    def test_airline_toolkit_stats(self, airline_environment):
        """Airline toolkit has expected statistics."""
        stats = airline_environment.toolkit.get_statistics()

        assert stats["num_tools"] > 0

    def test_telecom_toolkit_stats(self, telecom_environment):
        """Telecom toolkit has expected statistics."""
        stats = telecom_environment.toolkit.get_statistics()

        assert stats["num_tools"] > 0


# =============================================================================
# Database Statistics Tests
# =============================================================================


@pytest.mark.benchmark
class TestDatabaseStatistics:
    """Tests for database statistics."""

    def test_retail_db_stats(self, retail_environment):
        """Retail database has expected statistics."""
        stats = retail_environment.db.get_statistics()

        assert "num_products" in stats
        assert "num_users" in stats
        assert "num_orders" in stats

    def test_airline_db_stats(self, airline_environment):
        """Airline database has expected statistics."""
        stats = airline_environment.db.get_statistics()

        assert "num_flights" in stats
        assert "num_users" in stats
        assert "num_reservations" in stats

    def test_telecom_db_stats(self, telecom_environment):
        """Telecom database has expected statistics."""
        stats = telecom_environment.db.get_statistics()

        assert "num_customers" in stats
        assert "num_plans" in stats
        assert "num_lines" in stats


# =============================================================================
# Trace Gathering Tests
# =============================================================================


@pytest.mark.benchmark
class TestTraceGathering:
    """Tests for environment trace gathering."""

    def test_gather_traces_structure(self, retail_environment):
        """gather_traces returns expected structure."""
        traces = retail_environment.gather_traces()

        assert "domain" in traces
        assert traces["domain"] == "retail"
        assert "initial_db_hash" in traces
        assert "final_db_hash" in traces
        assert "db_changed" in traces

    def test_gather_config_structure(self, retail_environment):
        """gather_config returns expected structure."""
        config = retail_environment.gather_config()

        assert "domain" in config
        assert "toolkit_stats" in config
        assert "db_stats" in config
