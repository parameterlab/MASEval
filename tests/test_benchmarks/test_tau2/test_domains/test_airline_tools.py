"""Comprehensive tests for Tau2 airline domain tools.

Tests all airline toolkit functionality including read operations,
write operations (booking, cancellation, updates), and helper methods.
"""

import pytest

from maseval.benchmark.tau2.domains.base import ToolType


# =============================================================================
# Toolkit Infrastructure Tests
# =============================================================================


@pytest.mark.benchmark
class TestAirlineToolkitInfrastructure:
    """Tests for toolkit setup and infrastructure."""

    def test_toolkit_initialization(self, airline_toolkit):
        """Toolkit initializes with database."""
        assert airline_toolkit.db is not None
        assert len(airline_toolkit.tools) > 0

    def test_all_tools_callable(self, airline_toolkit):
        """All registered tools are callable."""
        for name, tool in airline_toolkit.tools.items():
            assert callable(tool), f"Tool {name} is not callable"

    def test_toolkit_statistics(self, airline_toolkit):
        """Toolkit reports correct statistics."""
        stats = airline_toolkit.get_statistics()
        assert stats["num_tools"] > 0
        assert stats["num_read_tools"] > 0
        assert stats["num_write_tools"] > 0

    def test_tool_descriptions_complete(self, airline_toolkit):
        """All tools have non-empty descriptions."""
        descriptions = airline_toolkit.get_tool_descriptions()
        for name, desc in descriptions.items():
            assert isinstance(desc, str) and len(desc) > 0, f"Tool {name} has no description"

    def test_database_hash_consistent(self, airline_toolkit):
        """Database hash is consistent for unchanged state."""
        hash1 = airline_toolkit.get_db_hash()
        hash2 = airline_toolkit.get_db_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256


# =============================================================================
# Tool Metadata Tests
# =============================================================================


@pytest.mark.benchmark
class TestAirlineToolMetadata:
    """Tests for tool metadata and type annotations."""

    def test_read_tools_marked_correctly(self, airline_toolkit):
        """Read-only tools have READ type."""
        read_tools = ["get_user_details", "get_reservation_details", "list_all_airports", "search_direct_flight"]
        for tool_name in read_tools:
            if airline_toolkit.has_tool(tool_name):
                assert airline_toolkit.tool_type(tool_name) == ToolType.READ

    def test_write_tools_marked_correctly(self, airline_toolkit):
        """State-modifying tools have WRITE type."""
        write_tools = ["book_reservation", "cancel_reservation", "update_reservation_baggages"]
        for tool_name in write_tools:
            if airline_toolkit.has_tool(tool_name):
                assert airline_toolkit.tool_type(tool_name) == ToolType.WRITE

    def test_tool_metadata_structure(self, airline_toolkit):
        """Tool metadata has required fields."""
        meta = airline_toolkit.get_tool_metadata("get_user_details")
        assert "description" in meta
        assert "inputs" in meta
        assert "tool_type" in meta
        assert "user_id" in meta["inputs"]

    def test_invalid_tool_metadata_raises(self, airline_toolkit):
        """Requesting metadata for nonexistent tool raises error."""
        with pytest.raises(ValueError, match="not found"):
            airline_toolkit.get_tool_metadata("nonexistent_tool")


# =============================================================================
# Read Tools - User Operations
# =============================================================================


@pytest.mark.benchmark
class TestAirlineUserOperations:
    """Tests for user-related read operations."""

    def test_get_user_details_valid(self, airline_toolkit):
        """Returns user details for valid user ID."""
        user_ids = list(airline_toolkit.db.users.keys())
        assert len(user_ids) > 0, "Test database has no users"

        user = airline_toolkit.use_tool("get_user_details", user_id=user_ids[0])
        assert user is not None
        assert user.user_id == user_ids[0]

    def test_get_user_details_invalid_raises(self, airline_toolkit):
        """Raises error for nonexistent user."""
        with pytest.raises(ValueError, match="not found"):
            airline_toolkit.use_tool("get_user_details", user_id="nonexistent_user_12345")


# =============================================================================
# Read Tools - Reservation Operations
# =============================================================================


@pytest.mark.benchmark
class TestAirlineReservationRead:
    """Tests for reservation read operations."""

    def test_get_reservation_details_valid(self, airline_toolkit):
        """Returns reservation details for valid ID."""
        reservation_ids = list(airline_toolkit.db.reservations.keys())
        assert len(reservation_ids) > 0, "Test database has no reservations"

        reservation = airline_toolkit.use_tool("get_reservation_details", reservation_id=reservation_ids[0])
        assert reservation is not None
        assert reservation.reservation_id == reservation_ids[0]

    def test_get_reservation_details_invalid_raises(self, airline_toolkit):
        """Raises error for nonexistent reservation."""
        with pytest.raises(ValueError, match="not found"):
            airline_toolkit.use_tool("get_reservation_details", reservation_id="INVALID")


# =============================================================================
# Read Tools - Flight Operations
# =============================================================================


@pytest.mark.benchmark
class TestAirlineFlightRead:
    """Tests for flight search and status operations."""

    def test_list_all_airports(self, airline_toolkit):
        """Returns list of airport codes."""
        airports = airline_toolkit.use_tool("list_all_airports")
        assert isinstance(airports, list)
        assert len(airports) > 0
        assert hasattr(airports[0], "iata") or hasattr(airports[0], "city")

    def test_search_direct_flight(self, airline_toolkit):
        """Searches for direct flights between cities."""
        result = airline_toolkit.use_tool(
            "search_direct_flight",
            origin="SFO",
            destination="JFK",
            date="2024-05-15",
        )
        assert isinstance(result, list)

    def test_search_onestop_flight(self, airline_toolkit):
        """Searches for one-stop connecting flights."""
        result = airline_toolkit.use_tool(
            "search_onestop_flight",
            origin="SFO",
            destination="MIA",
            date="2024-05-15",
        )
        assert isinstance(result, list)
        # Each result should be a pair of connecting flights
        for connection in result:
            assert isinstance(connection, list)

    def test_get_flight_status(self, airline_toolkit):
        """Returns flight status for valid flight/date."""
        flights = list(airline_toolkit.db.flights.keys())
        if not flights:
            pytest.skip("No flights in database")

        flight = airline_toolkit.db.flights[flights[0]]
        dates = list(flight.dates.keys())
        if not dates:
            pytest.skip("Flight has no dates")

        status = airline_toolkit.use_tool("get_flight_status", flight_number=flights[0], date=dates[0])
        assert isinstance(status, str)


# =============================================================================
# Generic Tools
# =============================================================================


@pytest.mark.benchmark
class TestAirlineGenericTools:
    """Tests for generic utility tools."""

    def test_calculate_simple_expression(self, airline_toolkit):
        """Evaluates simple arithmetic."""
        result = airline_toolkit.use_tool("calculate", expression="100 + 50")
        assert float(result) == 150.0

    def test_calculate_complex_expression(self, airline_toolkit):
        """Evaluates complex arithmetic with decimals."""
        result = airline_toolkit.use_tool("calculate", expression="(100 * 2.5) / 5")
        assert float(result) == 50.0

    def test_calculate_rejects_invalid_characters(self, airline_toolkit):
        """Rejects expressions with dangerous characters."""
        with pytest.raises(ValueError, match="Invalid characters"):
            airline_toolkit.use_tool("calculate", expression="__import__('os')")

    def test_transfer_to_human_agents(self, airline_toolkit):
        """Returns success for human transfer."""
        result = airline_toolkit.use_tool("transfer_to_human_agents", summary="Customer needs help")
        assert "Transfer successful" in result


# =============================================================================
# Write Tools - Reservation Cancellation
# =============================================================================


@pytest.mark.benchmark
class TestAirlineCancelReservation:
    """Tests for reservation cancellation."""

    def test_cancel_reservation_success(self, airline_toolkit):
        """Cancels active reservation and adds refund."""
        # Find an active (non-cancelled) reservation
        active_reservation = None
        for res_id, res in airline_toolkit.db.reservations.items():
            if res.status != "cancelled":
                active_reservation = res
                break

        if not active_reservation:
            pytest.skip("No active reservations to cancel")

        original_payment_count = len(active_reservation.payment_history)  # type: ignore[union-attr]
        result = airline_toolkit.use_tool("cancel_reservation", reservation_id=active_reservation.reservation_id)  # type: ignore[union-attr]

        assert result.status == "cancelled"
        assert len(result.payment_history) > original_payment_count  # Refund added

    def test_cancel_invalid_reservation_raises(self, airline_toolkit):
        """Raises error for nonexistent reservation."""
        with pytest.raises(ValueError, match="not found"):
            airline_toolkit.use_tool("cancel_reservation", reservation_id="INVALID123")


# =============================================================================
# Write Tools - Certificate Operations
# =============================================================================


@pytest.mark.benchmark
class TestAirlineSendCertificate:
    """Tests for sending certificates to users."""

    def test_send_certificate_success(self, airline_toolkit):
        """Sends certificate to valid user."""
        user_ids = list(airline_toolkit.db.users.keys())
        if not user_ids:
            pytest.skip("No users in database")

        user_id = user_ids[0]
        result = airline_toolkit.use_tool("send_certificate", user_id=user_id, amount=100)

        assert "Certificate" in result
        assert "added" in result

    def test_send_certificate_invalid_user_raises(self, airline_toolkit):
        """Raises error for nonexistent user."""
        with pytest.raises(ValueError, match="not found"):
            airline_toolkit.use_tool("send_certificate", user_id="invalid_user", amount=50)


# =============================================================================
# Write Tools - Baggage Updates
# =============================================================================


@pytest.mark.benchmark
class TestAirlineUpdateBaggages:
    """Tests for baggage update operations."""

    def test_update_baggages_success(self, airline_toolkit):
        """Updates baggage for valid reservation."""
        # Find a reservation with an associated user
        reservation = None
        user = None
        for res_id, res in airline_toolkit.db.reservations.items():
            if res.user_id in airline_toolkit.db.users:
                user = airline_toolkit.db.users[res.user_id]
                if user.payment_methods:
                    reservation = res
                    break

        if not reservation or not user:
            pytest.skip("No suitable reservation for baggage update")

        payment_id = list(user.payment_methods.keys())[0]  # type: ignore[union-attr]
        new_total = reservation.total_baggages + 1  # type: ignore[union-attr]
        new_nonfree = reservation.nonfree_baggages + 1  # type: ignore[union-attr]

        result = airline_toolkit.use_tool(
            "update_reservation_baggages",
            reservation_id=reservation.reservation_id,  # type: ignore[union-attr]
            total_baggages=new_total,
            nonfree_baggages=new_nonfree,
            payment_id=payment_id,
        )

        assert result.total_baggages == new_total
        assert result.nonfree_baggages == new_nonfree


# =============================================================================
# Write Tools - Flight Updates
# =============================================================================


@pytest.mark.benchmark
class TestAirlineUpdateFlights:
    """Tests for flight update operations."""

    def test_update_flights_keeps_existing(self, airline_toolkit):
        """Updating with same flights preserves them."""
        # Find a reservation with flights
        reservation = None
        user = None
        for res_id, res in airline_toolkit.db.reservations.items():
            if res.flights and res.user_id in airline_toolkit.db.users:
                user = airline_toolkit.db.users[res.user_id]
                if user.payment_methods:
                    reservation = res
                    break

        if not reservation or not user:
            pytest.skip("No suitable reservation for flight update")

        payment_id = list(user.payment_methods.keys())[0]  # type: ignore[union-attr]
        current_flights = [
            {"flight_number": f.flight_number, "date": f.date}
            for f in reservation.flights  # type: ignore[union-attr]
        ]

        result = airline_toolkit.use_tool(
            "update_reservation_flights",
            reservation_id=reservation.reservation_id,  # type: ignore[union-attr]
            cabin=reservation.cabin,  # type: ignore[union-attr]
            flights=current_flights,
            payment_id=payment_id,
        )

        assert len(result.flights) == len(current_flights)


# =============================================================================
# Write Tools - Passenger Updates
# =============================================================================


@pytest.mark.benchmark
class TestAirlineUpdatePassengers:
    """Tests for passenger update operations."""

    def test_update_passengers_success(self, airline_toolkit):
        """Updates passenger information."""
        # Find a reservation with passengers
        reservation = None
        for res_id, res in airline_toolkit.db.reservations.items():
            if res.passengers:
                reservation = res
                break

        if not reservation:
            pytest.skip("No reservation with passengers")

        # Create updated passenger list (same count, different names)
        updated_passengers = []
        for p in reservation.passengers:  # type: ignore[union-attr]
            updated_passengers.append(
                {
                    "first_name": p.first_name,
                    "last_name": "UpdatedName",
                    "dob": p.dob,
                }
            )

        result = airline_toolkit.use_tool(
            "update_reservation_passengers",
            reservation_id=reservation.reservation_id,  # type: ignore[union-attr]
            passengers=updated_passengers,
        )

        assert len(result.passengers) == len(updated_passengers)

    def test_update_passengers_count_mismatch_raises(self, airline_toolkit):
        """Raises error when passenger count doesn't match."""
        # Find a reservation with passengers
        reservation = None
        for res_id, res in airline_toolkit.db.reservations.items():
            if res.passengers:
                reservation = res
                break

        if not reservation:
            pytest.skip("No reservation with passengers")

        # Try to update with different number of passengers
        wrong_count_passengers = [{"first_name": "Test", "last_name": "User", "dob": "1990-01-01"}]

        if len(reservation.passengers) != 1:  # type: ignore[union-attr]
            with pytest.raises(ValueError, match="does not match"):
                airline_toolkit.use_tool(
                    "update_reservation_passengers",
                    reservation_id=reservation.reservation_id,  # type: ignore[union-attr]
                    passengers=wrong_count_passengers,
                )


# =============================================================================
# Write Tools - Booking
# =============================================================================


@pytest.mark.benchmark
class TestAirlineBookReservation:
    """Tests for reservation booking."""

    def test_book_reservation_validation(self, airline_toolkit):
        """Booking validates required fields."""
        user_ids = list(airline_toolkit.db.users.keys())
        if not user_ids:
            pytest.skip("No users in database")

        user_id = user_ids[0]
        user = airline_toolkit.db.users[user_id]

        if not user.payment_methods:
            pytest.skip("User has no payment methods")

        # Try booking with invalid payment method
        with pytest.raises(ValueError):
            airline_toolkit.use_tool(
                "book_reservation",
                user_id=user_id,
                origin="SFO",
                destination="JFK",
                flight_type="one_way",
                cabin="economy",
                flights=[{"flight_number": "INVALID", "date": "2024-05-15"}],
                passengers=[{"first_name": "Test", "last_name": "User", "dob": "1990-01-01"}],
                payment_methods=[{"payment_id": "invalid_payment", "amount": 100}],
                total_baggages=1,
                nonfree_baggages=0,
                insurance="no",
            )
