"""Unit tests for Tau2 telecom domain tools."""

import pytest

from maseval.benchmark.tau2.domains.base import ToolType
from maseval.benchmark.tau2.domains.telecom.models import LineStatus


# =============================================================================
# Toolkit Basic Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomToolkitBasic:
    """Basic tests for TelecomTools."""

    def test_toolkit_has_tools(self, telecom_toolkit):
        """Toolkit has tools available."""
        assert len(telecom_toolkit.tools) > 0

    def test_all_tools_callable(self, telecom_toolkit):
        """All tools are callable methods."""
        for name, tool in telecom_toolkit.tools.items():
            assert callable(tool), f"Tool {name} is not callable"

    def test_toolkit_statistics(self, telecom_toolkit):
        """Toolkit provides statistics."""
        stats = telecom_toolkit.get_statistics()

        assert stats["num_tools"] > 0
        assert stats["num_read_tools"] >= 0
        assert stats["num_write_tools"] >= 0

    def test_toolkit_descriptions(self, telecom_toolkit):
        """Toolkit provides tool descriptions."""
        descriptions = telecom_toolkit.get_tool_descriptions()

        assert len(descriptions) > 0
        for name, desc in descriptions.items():
            assert isinstance(desc, str)


# =============================================================================
# Read Tool Tests - Customer Lookup
# =============================================================================


@pytest.mark.benchmark
class TestTelecomCustomerLookup:
    """Tests for telecom customer lookup tools."""

    def test_get_customer_by_id(self, telecom_toolkit):
        """get_customer_by_id returns customer information."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in test database")

        customer = customers[0]
        result = telecom_toolkit.use_tool("get_customer_by_id", customer_id=customer.customer_id)

        assert result is not None
        assert result.customer_id == customer.customer_id

    def test_get_customer_by_id_invalid(self, telecom_toolkit):
        """get_customer_by_id raises for invalid customer."""
        with pytest.raises(ValueError, match="not found"):
            telecom_toolkit.use_tool("get_customer_by_id", customer_id="INVALID_ID")

    def test_get_customer_by_phone(self, telecom_toolkit):
        """get_customer_by_phone returns customer information."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in test database")

        customer = customers[0]
        result = telecom_toolkit.use_tool("get_customer_by_phone", phone_number=customer.phone_number)

        assert result is not None
        assert result.customer_id == customer.customer_id

    def test_get_customer_by_phone_invalid(self, telecom_toolkit):
        """get_customer_by_phone raises for invalid phone."""
        with pytest.raises(ValueError, match="not found"):
            telecom_toolkit.use_tool("get_customer_by_phone", phone_number="000-000-0000")

    def test_get_customer_by_name(self, telecom_toolkit):
        """get_customer_by_name returns matching customers."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in test database")

        customer = customers[0]
        result = telecom_toolkit.use_tool(
            "get_customer_by_name",
            full_name=customer.full_name,
            dob=customer.date_of_birth,
        )

        assert result is not None
        assert isinstance(result, list)
        if len(result) > 0:
            assert result[0].customer_id == customer.customer_id


# =============================================================================
# Read Tool Tests - Billing
# =============================================================================


@pytest.mark.benchmark
class TestTelecomBillingRead:
    """Tests for telecom billing read tools."""

    def test_get_bills_for_customer(self, telecom_toolkit):
        """get_bills_for_customer returns bills."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in test database")

        customer = customers[0]
        if not customer.bill_ids:
            pytest.skip("Customer has no bills")

        result = telecom_toolkit.use_tool("get_bills_for_customer", customer_id=customer.customer_id)

        assert result is not None
        assert isinstance(result, list)

    def test_get_data_usage(self, telecom_toolkit):
        """get_data_usage returns usage information."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in test database")

        customer = customers[0]
        if not customer.line_ids:
            pytest.skip("Customer has no lines")

        result = telecom_toolkit.use_tool(
            "get_data_usage",
            customer_id=customer.customer_id,
            line_id=customer.line_ids[0],
        )

        assert result is not None
        assert "data_used_gb" in result
        assert "data_limit_gb" in result


# =============================================================================
# Read Tool Tests - Details by ID
# =============================================================================


@pytest.mark.benchmark
class TestTelecomDetailsById:
    """Tests for get_details_by_id tool."""

    def test_get_details_by_customer_id(self, telecom_toolkit):
        """get_details_by_id returns customer for C-prefixed ID."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in test database")

        customer = customers[0]
        result = telecom_toolkit.use_tool("get_details_by_id", id=customer.customer_id)

        assert result is not None
        assert result.customer_id == customer.customer_id

    def test_get_details_by_line_id(self, telecom_toolkit):
        """get_details_by_id returns line for L-prefixed ID."""
        lines = telecom_toolkit.db.lines
        if not lines:
            pytest.skip("No lines in test database")

        line = lines[0]
        result = telecom_toolkit.use_tool("get_details_by_id", id=line.line_id)

        assert result is not None
        assert result.line_id == line.line_id

    def test_get_details_by_invalid_id(self, telecom_toolkit):
        """get_details_by_id raises for invalid ID format."""
        with pytest.raises(ValueError, match="Unknown ID format"):
            telecom_toolkit.use_tool("get_details_by_id", id="INVALID_123")


# =============================================================================
# Tool Type Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomToolTypes:
    """Tests for telecom tool type annotations."""

    def test_read_tools_have_correct_type(self, telecom_toolkit):
        """Read tools are marked as READ type."""
        read_tools = ["get_customer_by_id", "get_customer_by_phone", "get_bills_for_customer", "get_data_usage"]

        for tool_name in read_tools:
            if telecom_toolkit.has_tool(tool_name):
                tool_type = telecom_toolkit.tool_type(tool_name)
                assert tool_type == ToolType.READ, f"{tool_name} should be READ type"

    def test_write_tools_have_correct_type(self, telecom_toolkit):
        """Write tools are marked as WRITE type."""
        write_tools = ["suspend_line", "resume_line", "enable_roaming", "disable_roaming", "refuel_data"]

        for tool_name in write_tools:
            if telecom_toolkit.has_tool(tool_name):
                tool_type = telecom_toolkit.tool_type(tool_name)
                assert tool_type == ToolType.WRITE, f"{tool_name} should be WRITE type"


# =============================================================================
# Database Hash Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomDatabaseHash:
    """Tests for telecom database hashing."""

    def test_hash_consistent(self, telecom_toolkit):
        """Database hash is consistent."""
        hash1 = telecom_toolkit.get_db_hash()
        hash2 = telecom_toolkit.get_db_hash()

        assert hash1 == hash2

    def test_hash_is_sha256(self, telecom_toolkit):
        """Database hash is SHA-256 format."""
        hash_value = telecom_toolkit.get_db_hash()

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in hash_value)


# =============================================================================
# Write Tool Tests - Line Management
# =============================================================================


@pytest.mark.benchmark
class TestTelecomLineManagement:
    """Tests for line management tools."""

    def test_suspend_active_line(self, telecom_toolkit):
        """Successfully suspend an active line."""
        # Find an active line
        active_line = None
        customer_id = None
        for customer in telecom_toolkit.db.customers:
            for line_id in customer.line_ids:
                line = telecom_toolkit._get_line_by_id(line_id)
                if line.status == LineStatus.ACTIVE:
                    active_line = line
                    customer_id = customer.customer_id
                    break
            if active_line:
                break

        if not active_line:
            pytest.skip("No active lines in test database")

        result = telecom_toolkit.use_tool(
            "suspend_line",
            customer_id=customer_id,
            line_id=active_line.line_id,  # type: ignore[union-attr]
            reason="Customer requested suspension",
        )

        assert result is not None
        assert result["line"].status == LineStatus.SUSPENDED

    def test_suspend_non_active_line_fails(self, telecom_toolkit):
        """Cannot suspend a non-active line."""
        # Find a suspended line
        suspended_line = None
        customer_id = None
        for customer in telecom_toolkit.db.customers:
            for line_id in customer.line_ids:
                line = telecom_toolkit._get_line_by_id(line_id)
                if line.status == LineStatus.SUSPENDED:
                    suspended_line = line
                    customer_id = customer.customer_id
                    break
            if suspended_line:
                break

        if not suspended_line:
            pytest.skip("No suspended lines in test database")

        with pytest.raises(ValueError, match="must be active"):
            telecom_toolkit.use_tool(
                "suspend_line",
                customer_id=customer_id,
                line_id=suspended_line.line_id,  # type: ignore[union-attr]
                reason="Test",
            )

    def test_resume_suspended_line(self, telecom_toolkit):
        """Successfully resume a suspended line."""
        # Find a suspended line
        suspended_line = None
        customer_id = None
        for customer in telecom_toolkit.db.customers:
            for line_id in customer.line_ids:
                line = telecom_toolkit._get_line_by_id(line_id)
                if line.status == LineStatus.SUSPENDED:
                    suspended_line = line
                    customer_id = customer.customer_id
                    break
            if suspended_line:
                break

        if not suspended_line:
            pytest.skip("No suspended lines in test database")

        result = telecom_toolkit.use_tool(
            "resume_line",
            customer_id=customer_id,
            line_id=suspended_line.line_id,  # type: ignore[union-attr]
        )

        assert result is not None
        assert result["line"].status == LineStatus.ACTIVE


# =============================================================================
# Write Tool Tests - Roaming
# =============================================================================


@pytest.mark.benchmark
class TestTelecomRoaming:
    """Tests for roaming tools."""

    def test_enable_roaming(self, telecom_toolkit):
        """Enable roaming on a line."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in test database")

        customer = customers[0]
        if not customer.line_ids:
            pytest.skip("Customer has no lines")

        line_id = customer.line_ids[0]
        line = telecom_toolkit._get_line_by_id(line_id)

        # First disable if enabled
        if line.roaming_enabled:
            telecom_toolkit.use_tool(
                "disable_roaming",
                customer_id=customer.customer_id,
                line_id=line_id,
            )

        result = telecom_toolkit.use_tool(
            "enable_roaming",
            customer_id=customer.customer_id,
            line_id=line_id,
        )

        assert "enabled" in result.lower() or "already" in result.lower()
        assert line.roaming_enabled is True

    def test_disable_roaming(self, telecom_toolkit):
        """Disable roaming on a line."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in test database")

        customer = customers[0]
        if not customer.line_ids:
            pytest.skip("Customer has no lines")

        line_id = customer.line_ids[0]
        line = telecom_toolkit._get_line_by_id(line_id)

        # First enable if disabled
        if not line.roaming_enabled:
            telecom_toolkit.use_tool(
                "enable_roaming",
                customer_id=customer.customer_id,
                line_id=line_id,
            )

        result = telecom_toolkit.use_tool(
            "disable_roaming",
            customer_id=customer.customer_id,
            line_id=line_id,
        )

        assert "disabled" in result.lower() or "already" in result.lower()
        assert line.roaming_enabled is False


# =============================================================================
# Write Tool Tests - Data Refueling
# =============================================================================


@pytest.mark.benchmark
class TestTelecomDataRefueling:
    """Tests for data refueling tools."""

    def test_refuel_data(self, telecom_toolkit):
        """Successfully refuel data for a line."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in test database")

        customer = customers[0]
        if not customer.line_ids:
            pytest.skip("Customer has no lines")

        line_id = customer.line_ids[0]
        line = telecom_toolkit._get_line_by_id(line_id)
        original_data = line.data_refueling_gb

        result = telecom_toolkit.use_tool(
            "refuel_data",
            customer_id=customer.customer_id,
            line_id=line_id,
            gb_amount=5.0,
        )

        assert result is not None
        assert result["new_data_refueling_gb"] == original_data + 5.0
        assert result["charge"] > 0

    def test_refuel_data_invalid_amount(self, telecom_toolkit):
        """Refuel with invalid amount fails."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in test database")

        customer = customers[0]
        if not customer.line_ids:
            pytest.skip("Customer has no lines")

        with pytest.raises(ValueError, match="positive"):
            telecom_toolkit.use_tool(
                "refuel_data",
                customer_id=customer.customer_id,
                line_id=customer.line_ids[0],
                gb_amount=-5.0,
            )


# =============================================================================
# Generic Tool Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomGenericTools:
    """Tests for generic tools."""

    def test_transfer_to_human(self, telecom_toolkit):
        """Transfer to human agent returns success."""
        result = telecom_toolkit.use_tool(
            "transfer_to_human_agents",
            summary="Customer needs technical support",
        )
        assert "Transfer successful" in result


# =============================================================================
# Tool Metadata Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomToolMetadata:
    """Tests for telecom tool metadata."""

    def test_get_tool_metadata(self, telecom_toolkit):
        """get_tool_metadata returns expected structure."""
        meta = telecom_toolkit.get_tool_metadata("get_customer_by_id")

        assert "description" in meta
        assert "inputs" in meta
        assert "tool_type" in meta
        assert meta["tool_type"] == ToolType.READ

    def test_get_tool_metadata_invalid_tool(self, telecom_toolkit):
        """get_tool_metadata raises for invalid tool."""
        with pytest.raises(ValueError, match="not found"):
            telecom_toolkit.get_tool_metadata("nonexistent_tool")

    def test_tool_inputs_extracted(self, telecom_toolkit):
        """Tool inputs are extracted from signature."""
        meta = telecom_toolkit.get_tool_metadata("get_customer_by_id")

        assert "customer_id" in meta["inputs"]
        assert meta["inputs"]["customer_id"]["type"] == "string"


# =============================================================================
# Read Tool Tests - Bill Details
# =============================================================================


@pytest.mark.benchmark
class TestTelecomBillDetails:
    """Tests for bill detail retrieval."""

    def test_get_bill_by_id(self, telecom_toolkit):
        """Get bill by ID returns bill details."""
        bills = telecom_toolkit.db.bills
        if not bills:
            pytest.skip("No bills in database")

        bill = bills[0]
        result = telecom_toolkit.use_tool("get_details_by_id", id=bill.bill_id)

        assert result is not None
        assert result.bill_id == bill.bill_id

    def test_get_plan_by_id(self, telecom_toolkit):
        """Get plan by ID returns plan details."""
        plans = telecom_toolkit.db.plans
        if not plans:
            pytest.skip("No plans in database")

        plan = plans[0]
        result = telecom_toolkit.use_tool("get_details_by_id", id=plan.plan_id)

        assert result is not None
        assert result.plan_id == plan.plan_id


# =============================================================================
# Write Tool Tests - Plan Changes
# =============================================================================


@pytest.mark.benchmark
class TestTelecomPlanChanges:
    """Tests for plan change operations."""

    def test_change_plan_validation(self, telecom_toolkit):
        """Plan change validates plan exists."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in database")

        customer = customers[0]
        if not customer.line_ids:
            pytest.skip("Customer has no lines")

        with pytest.raises(ValueError):
            telecom_toolkit.use_tool(
                "change_plan",
                customer_id=customer.customer_id,
                line_id=customer.line_ids[0],
                new_plan_id="INVALID_PLAN",
            )


# =============================================================================
# Write Tool Tests - Payment Processing
# =============================================================================


@pytest.mark.benchmark
class TestTelecomPayment:
    """Tests for payment processing."""

    def test_process_payment_validation(self, telecom_toolkit):
        """Payment processing validates bill exists."""
        customers = telecom_toolkit.db.customers
        if not customers:
            pytest.skip("No customers in database")

        customer = customers[0]

        with pytest.raises(ValueError):
            telecom_toolkit.use_tool(
                "process_payment",
                customer_id=customer.customer_id,
                bill_id="INVALID_BILL",
                amount=100.0,
            )


# =============================================================================
# Write Tool Tests - Line Activation
# =============================================================================


@pytest.mark.benchmark
class TestTelecomLineActivation:
    """Tests for line activation tools."""

    def test_resume_non_suspended_line_fails(self, telecom_toolkit):
        """Cannot resume a line that is not suspended."""
        active_line = None
        customer_id = None
        for customer in telecom_toolkit.db.customers:
            for line_id in customer.line_ids:
                line = telecom_toolkit._get_line_by_id(line_id)
                if line.status == LineStatus.ACTIVE:
                    active_line = line
                    customer_id = customer.customer_id
                    break
            if active_line:
                break

        if not active_line:
            pytest.skip("No active lines in database")

        with pytest.raises(ValueError, match="suspended"):
            telecom_toolkit.use_tool(
                "resume_line",
                customer_id=customer_id,
                line_id=active_line.line_id,  # type: ignore[union-attr]
            )


# =============================================================================
# Read Tool Tests - Service Status
# =============================================================================


@pytest.mark.benchmark
class TestTelecomServiceStatus:
    """Tests for service status tools."""

    def test_get_service_outages(self, telecom_toolkit):
        """Get service outages returns outage list."""
        if telecom_toolkit.has_tool("get_service_outages"):
            result = telecom_toolkit.use_tool("get_service_outages")
            assert isinstance(result, list)

    def test_check_network_coverage(self, telecom_toolkit):
        """Check network coverage for area."""
        if telecom_toolkit.has_tool("check_network_coverage"):
            result = telecom_toolkit.use_tool("check_network_coverage", zip_code="94102")
            assert result is not None
