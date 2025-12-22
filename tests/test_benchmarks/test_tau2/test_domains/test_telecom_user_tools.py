"""Unit tests for Tau2 telecom user tools."""

import pytest
from maseval.benchmark.tau2.domains.telecom.user_models import SimStatus, NetworkStatus, NetworkModePreference


# =============================================================================
# Toolkit Basic Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomUserToolkitBasic:
    """Basic tests for TelecomUserTools."""

    def test_toolkit_has_tools(self, telecom_user_toolkit):
        """Toolkit has tools available."""
        assert len(telecom_user_toolkit.tools) > 0

    def test_all_tools_callable(self, telecom_user_toolkit):
        """All tools are callable methods."""
        for name, tool in telecom_user_toolkit.tools.items():
            assert callable(tool), f"Tool {name} is not callable"

    def test_toolkit_statistics(self, telecom_user_toolkit):
        """Toolkit provides statistics."""
        stats = telecom_user_toolkit.get_statistics()
        assert stats["num_tools"] > 0
        assert stats["num_read_tools"] >= 0
        assert stats["num_write_tools"] >= 0

    def test_toolkit_descriptions(self, telecom_user_toolkit):
        """Toolkit provides tool descriptions."""
        descriptions = telecom_user_toolkit.get_tool_descriptions()
        assert len(descriptions) > 0
        for name, desc in descriptions.items():
            assert isinstance(desc, str)

    def test_user_db_initialized(self, telecom_user_toolkit):
        """User DB is initialized when toolkit is created."""
        assert telecom_user_toolkit.db.user_db is not None
        assert telecom_user_toolkit.db.user_db.device is not None
        assert telecom_user_toolkit.db.user_db.surroundings is not None


# =============================================================================
# Read Tool Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomUserReadTools:
    """Tests for telecom user read-only tools."""

    def test_check_status_bar(self, telecom_user_toolkit):
        """check_status_bar returns correct structure."""
        status = telecom_user_toolkit.use_tool("check_status_bar")
        assert "signal_strength" in status
        assert "network_type" in status
        assert "wifi_connected" in status
        assert "airplane_mode" in status
        assert "battery_level" in status

    def test_check_network_status(self, telecom_user_toolkit):
        """check_network_status returns correct structure."""
        status = telecom_user_toolkit.use_tool("check_network_status")
        assert "status" in status
        assert "technology" in status
        assert "roaming" in status
        assert "mobile_data" in status

    def test_check_installed_apps(self, telecom_user_toolkit):
        """check_installed_apps returns list."""
        apps = telecom_user_toolkit.use_tool("check_installed_apps")
        assert isinstance(apps, list)

    def test_check_apn_settings(self, telecom_user_toolkit):
        """check_apn_settings returns dict."""
        apn = telecom_user_toolkit.use_tool("check_apn_settings")
        assert isinstance(apn, dict)
        assert "name" in apn
        assert "apn" in apn


# =============================================================================
# Write Tool Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomUserWriteTools:
    """Tests for telecom user state-modifying tools."""

    def test_toggle_airplane_mode(self, telecom_user_toolkit):
        """toggle_airplane_mode changes state."""
        # Enable
        telecom_user_toolkit.use_tool("toggle_airplane_mode", enable=True)
        assert telecom_user_toolkit.db.user_db.device.airplane_mode is True
        assert telecom_user_toolkit.db.user_db.device.network_status == NetworkStatus.NO_SERVICE

        # Disable
        telecom_user_toolkit.use_tool("toggle_airplane_mode", enable=False)
        assert telecom_user_toolkit.db.user_db.device.airplane_mode is False
        # Should reconnect if SIM is active
        if telecom_user_toolkit.db.user_db.device.sim_status == SimStatus.ACTIVE:
            assert telecom_user_toolkit.db.user_db.device.network_status == NetworkStatus.CONNECTED

    def test_toggle_data(self, telecom_user_toolkit):
        """toggle_data changes state."""
        telecom_user_toolkit.use_tool("toggle_data", enable=False)
        assert telecom_user_toolkit.db.user_db.device.mobile_data_enabled is False

        telecom_user_toolkit.use_tool("toggle_data", enable=True)
        assert telecom_user_toolkit.db.user_db.device.mobile_data_enabled is True

    def test_set_network_mode_preference(self, telecom_user_toolkit):
        """set_network_mode_preference changes state."""
        pref = NetworkModePreference.FOUR_G_ONLY.value
        result = telecom_user_toolkit.use_tool("set_network_mode_preference", preference=pref)

        assert telecom_user_toolkit.db.user_db.device.network_mode_preference == NetworkModePreference.FOUR_G_ONLY
        assert "set to" in result

    def test_set_network_mode_invalid(self, telecom_user_toolkit):
        """set_network_mode_preference handles invalid input."""
        result = telecom_user_toolkit.use_tool("set_network_mode_preference", preference="invalid_mode")
        assert "Invalid preference" in result

    def test_reboot_device(self, telecom_user_toolkit):
        """reboot_device resets ephemeral state."""
        # Set reset_at_reboot = True
        telecom_user_toolkit.db.user_db.device.apn_settings.reset_at_reboot = True
        telecom_user_toolkit.db.user_db.device.apn_settings.apn = "custom_apn"

        telecom_user_toolkit.use_tool("reboot_device")

        # Should have reset to default
        assert telecom_user_toolkit.db.user_db.device.apn_settings.apn == "internet"
