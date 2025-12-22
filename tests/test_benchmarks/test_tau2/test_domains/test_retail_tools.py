"""Unit tests for Tau2 retail domain tools."""

import pytest

from maseval.benchmark.tau2.domains.base import ToolType


# =============================================================================
# Toolkit Basic Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailToolkitBasic:
    """Basic tests for RetailTools."""

    def test_toolkit_has_tools(self, retail_toolkit):
        """Toolkit has tools available."""
        assert len(retail_toolkit.tools) > 0

    def test_all_tools_callable(self, retail_toolkit):
        """All tools are callable methods."""
        for name, tool in retail_toolkit.tools.items():
            assert callable(tool), f"Tool {name} is not callable"

    def test_toolkit_statistics(self, retail_toolkit):
        """Toolkit provides statistics."""
        stats = retail_toolkit.get_statistics()

        assert stats["num_tools"] > 0
        assert stats["num_read_tools"] >= 0
        assert stats["num_write_tools"] >= 0

    def test_toolkit_descriptions(self, retail_toolkit):
        """Toolkit provides tool descriptions."""
        descriptions = retail_toolkit.get_tool_descriptions()

        assert len(descriptions) > 0
        for name, desc in descriptions.items():
            assert isinstance(desc, str)


# =============================================================================
# Read Tool Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailReadTools:
    """Tests for retail read-only tools."""

    def test_get_user_details(self, retail_toolkit):
        """get_user_details returns user information."""
        users = list(retail_toolkit.db.users.keys())
        if not users:
            pytest.skip("No users in test database")

        user_id = users[0]
        result = retail_toolkit.use_tool("get_user_details", user_id=user_id)

        assert result is not None
        assert hasattr(result, "user_id") or "user_id" in str(result)

    def test_get_user_details_invalid_user(self, retail_toolkit):
        """get_user_details raises for invalid user."""
        with pytest.raises(ValueError, match="not found"):
            retail_toolkit.use_tool("get_user_details", user_id="nonexistent_user")

    def test_get_order_details(self, retail_toolkit):
        """get_order_details returns order information."""
        orders = list(retail_toolkit.db.orders.keys())
        if not orders:
            pytest.skip("No orders in test database")

        order_id = orders[0]
        result = retail_toolkit.use_tool("get_order_details", order_id=order_id)

        assert result is not None

    def test_get_order_details_invalid_order(self, retail_toolkit):
        """get_order_details raises for invalid order."""
        with pytest.raises(ValueError, match="not found"):
            retail_toolkit.use_tool("get_order_details", order_id="nonexistent_order")

    def test_list_all_product_types(self, retail_toolkit):
        """list_all_product_types returns product categories."""
        result = retail_toolkit.use_tool("list_all_product_types")

        assert result is not None
        assert isinstance(result, (list, dict, str))

    def test_get_product_details(self, retail_toolkit):
        """get_product_details returns product information."""
        products = list(retail_toolkit.db.products.keys())
        if not products:
            pytest.skip("No products in test database")

        product_id = products[0]
        result = retail_toolkit.use_tool("get_product_details", product_id=product_id)

        assert result is not None


# =============================================================================
# Tool Type Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailToolTypes:
    """Tests for retail tool type annotations."""

    def test_read_tools_have_correct_type(self, retail_toolkit):
        """Read tools are marked as READ type."""
        read_tools = ["get_user_details", "get_order_details", "list_all_product_types", "get_product_details"]

        for tool_name in read_tools:
            if retail_toolkit.has_tool(tool_name):
                tool_type = retail_toolkit.tool_type(tool_name)
                assert tool_type == ToolType.READ, f"{tool_name} should be READ type"

    def test_write_tools_have_correct_type(self, retail_toolkit):
        """Write tools are marked as WRITE type."""
        write_tools = ["cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address"]

        for tool_name in write_tools:
            if retail_toolkit.has_tool(tool_name):
                tool_type = retail_toolkit.tool_type(tool_name)
                assert tool_type == ToolType.WRITE, f"{tool_name} should be WRITE type"


# =============================================================================
# Database Hash Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailDatabaseHash:
    """Tests for retail database hashing."""

    def test_hash_consistent(self, retail_toolkit):
        """Database hash is consistent."""
        hash1 = retail_toolkit.get_db_hash()
        hash2 = retail_toolkit.get_db_hash()

        assert hash1 == hash2

    def test_hash_is_sha256(self, retail_toolkit):
        """Database hash is SHA-256 format."""
        hash_value = retail_toolkit.get_db_hash()

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in hash_value)


# =============================================================================
# Tool Metadata Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailToolMetadata:
    """Tests for retail tool metadata."""

    def test_get_tool_metadata(self, retail_toolkit):
        """get_tool_metadata returns expected structure."""
        meta = retail_toolkit.get_tool_metadata("get_user_details")

        assert "description" in meta
        assert "inputs" in meta
        assert "tool_type" in meta
        assert meta["tool_type"] == ToolType.READ

    def test_get_tool_metadata_invalid_tool(self, retail_toolkit):
        """get_tool_metadata raises for invalid tool."""
        with pytest.raises(ValueError, match="not found"):
            retail_toolkit.get_tool_metadata("nonexistent_tool")

    def test_tool_inputs_extracted(self, retail_toolkit):
        """Tool inputs are extracted from signature."""
        meta = retail_toolkit.get_tool_metadata("get_user_details")

        assert "user_id" in meta["inputs"]
        assert meta["inputs"]["user_id"]["type"] == "string"
