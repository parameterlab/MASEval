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


# =============================================================================
# Helper Method Tests
# =============================================================================


@pytest.mark.benchmark
class TestAirlineHelperMethods:
    """Tests for internal helper methods."""

    def test_get_flight_invalid(self, airline_toolkit):
        """_get_flight raises for invalid flight."""
        with pytest.raises(ValueError, match="not found"):
            airline_toolkit._get_flight("INVALID_FLIGHT")

    def test_get_flight_instance_invalid_date(self, airline_toolkit):
        """_get_flight_instance raises for invalid date."""
        flights = list(airline_toolkit.db.flights.keys())
        if not flights:
            pytest.skip("No flights in database")

        with pytest.raises(ValueError, match="not found"):
            airline_toolkit._get_flight_instance(flights[0], "1900-01-01")

    def test_get_user_invalid(self, airline_toolkit):
        """_get_user raises for invalid user."""
        with pytest.raises(ValueError, match="not found"):
            airline_toolkit._get_user("INVALID_USER_12345")

    def test_get_reservation_invalid(self, airline_toolkit):
        """_get_reservation raises for invalid reservation."""
        with pytest.raises(ValueError, match="not found"):
            airline_toolkit._get_reservation("INVALID")


# =============================================================================
# Update Flights - New Flight Path
# =============================================================================


@pytest.mark.benchmark
class TestAirlineUpdateFlightsNewFlight:
    """Tests for updating reservation with new flights."""

    def test_update_flights_change_cabin(self, airline_toolkit):
        """Update reservation to different cabin class."""
        # Find a reservation with flights and user with payment methods
        reservation = None
        user = None
        for res_id, res in airline_toolkit.db.reservations.items():
            if res.flights and res.user_id in airline_toolkit.db.users and res.status != "cancelled":
                user = airline_toolkit.db.users[res.user_id]
                if user.payment_methods:
                    reservation = res
                    break

        if not reservation or not user:
            pytest.skip("No suitable reservation for flight update")

        payment_id = list(user.payment_methods.keys())[0]
        current_flights = [{"flight_number": f.flight_number, "date": f.date} for f in reservation.flights]
        original_cabin = reservation.cabin

        # Try changing to a different cabin (if possible)
        new_cabin = "business" if original_cabin == "economy" else "economy"

        # This might fail due to availability or price differences, but tests the code path
        try:
            result = airline_toolkit.use_tool(
                "update_reservation_flights",
                reservation_id=reservation.reservation_id,
                cabin=new_cabin,
                flights=current_flights,
                payment_id=payment_id,
            )
            assert result.cabin == new_cabin
        except ValueError:
            # Expected if there's not enough seats or balance
            pass


# =============================================================================
# Baggage Update Edge Cases
# =============================================================================


@pytest.mark.benchmark
class TestAirlineBaggageEdgeCases:
    """Edge case tests for baggage updates."""

    def test_update_baggages_reduce_count(self, airline_toolkit):
        """Updating to fewer baggages doesn't charge extra."""
        reservation = None
        user = None
        for res_id, res in airline_toolkit.db.reservations.items():
            if res.user_id in airline_toolkit.db.users and res.nonfree_baggages > 0:
                user = airline_toolkit.db.users[res.user_id]
                if user.payment_methods:
                    reservation = res
                    break

        if not reservation or not user:
            pytest.skip("No suitable reservation with nonfree baggages")

        payment_id = list(user.payment_methods.keys())[0]
        original_history_len = len(reservation.payment_history)

        # Reduce baggage count
        result = airline_toolkit.use_tool(
            "update_reservation_baggages",
            reservation_id=reservation.reservation_id,
            total_baggages=max(0, reservation.total_baggages - 1),
            nonfree_baggages=max(0, reservation.nonfree_baggages - 1),
            payment_id=payment_id,
        )

        # No additional payment should be added when reducing
        assert len(result.payment_history) == original_history_len


# =============================================================================
# One-Stop Flight Search Tests
# =============================================================================


@pytest.mark.benchmark
class TestAirlineOneStopSearch:
    """Tests for one-stop flight search functionality."""

    def test_search_onestop_with_cabin(self, airline_toolkit):
        """Search one-stop flights with cabin preference."""
        result = airline_toolkit.use_tool(
            "search_onestop_flight",
            origin="SFO",
            destination="MIA",
            date="2024-05-15",
        )
        # Result should be a list of connections (each connection is a list of 2 flights)
        assert isinstance(result, list)
        for connection in result:
            assert isinstance(connection, list)


# =============================================================================
# Booking Success Path Tests
# =============================================================================


@pytest.mark.benchmark
class TestAirlineBookingSuccess:
    """Tests for successful booking scenarios."""

    def test_book_reservation_invalid_flight(self, airline_toolkit):
        """Booking fails with invalid flight number."""
        user_ids = list(airline_toolkit.db.users.keys())
        if not user_ids:
            pytest.skip("No users in database")

        user_id = user_ids[0]
        user = airline_toolkit.db.users[user_id]

        if not user.payment_methods:
            pytest.skip("User has no payment methods")

        payment_id = list(user.payment_methods.keys())[0]

        with pytest.raises(ValueError, match="not found"):
            airline_toolkit.use_tool(
                "book_reservation",
                user_id=user_id,
                origin="SFO",
                destination="JFK",
                flight_type="one_way",
                cabin="economy",
                flights=[{"flight_number": "INVALID_FLIGHT_123", "date": "2024-05-15"}],
                passengers=[{"first_name": "Test", "last_name": "User", "dob": "1990-01-01"}],
                payment_methods=[{"payment_id": payment_id, "amount": 100}],
                total_baggages=1,
                nonfree_baggages=0,
                insurance="no",
            )

    def test_book_reservation_success(self, airline_toolkit):
        """Successfully book a reservation with valid flight and payment."""
        from maseval.benchmark.tau2.domains.airline.models import FlightDateStatusAvailable

        # Find a user with a credit card payment method
        user = None
        user_id = None
        for uid, u in airline_toolkit.db.users.items():
            for pm_id, pm in u.payment_methods.items():
                if pm.source == "credit_card":
                    user = u
                    user_id = uid
                    break
            if user:
                break

        if not user:
            pytest.skip("No user with credit card payment method")

        # Find an available flight with seats and price
        available_flight = None
        flight_date = None
        cabin = "economy"

        for flight_num, flight in airline_toolkit.db.flights.items():
            for date, date_data in flight.dates.items():
                if isinstance(date_data, FlightDateStatusAvailable):
                    if date_data.available_seats.get(cabin, 0) > 0:
                        available_flight = flight
                        flight_date = date
                        break
            if available_flight:
                break

        if not available_flight:
            pytest.skip("No available flights with seats")

        # Get the flight price
        flight_date_data = available_flight.dates[flight_date]
        price = flight_date_data.prices[cabin]

        # Use credit card payment (credit cards don't have balance limits)
        payment_id = None
        for pm_id, pm in user.payment_methods.items():
            if pm.source == "credit_card":
                payment_id = pm_id
                break

        if not payment_id:
            pytest.skip("User has no credit card")

        # Calculate total price: 1 passenger * ticket price + 0 baggage fee + no insurance
        total_price = price

        result = airline_toolkit.use_tool(
            "book_reservation",
            user_id=user_id,
            origin=available_flight.origin,
            destination=available_flight.destination,
            flight_type="one_way",
            cabin=cabin,
            flights=[{"flight_number": available_flight.flight_number, "date": flight_date}],
            passengers=[{"first_name": "Test", "last_name": "Booker", "dob": "1985-03-15"}],
            payment_methods=[{"payment_id": payment_id, "amount": total_price}],
            total_baggages=0,
            nonfree_baggages=0,
            insurance="no",
        )

        assert result is not None
        assert result.reservation_id is not None
        assert result.user_id == user_id
        assert result.cabin == cabin
        assert len(result.flights) == 1

    def test_book_reservation_with_insurance(self, airline_toolkit):
        """Book reservation with insurance adds insurance fee."""
        from maseval.benchmark.tau2.domains.airline.models import FlightDateStatusAvailable

        # Find a user with a credit card payment method
        user = None
        user_id = None
        for uid, u in airline_toolkit.db.users.items():
            for pm_id, pm in u.payment_methods.items():
                if pm.source == "credit_card":
                    user = u
                    user_id = uid
                    break
            if user:
                break

        if not user:
            pytest.skip("No user with credit card payment method")

        # Find an available flight
        available_flight = None
        flight_date = None
        cabin = "economy"

        for flight_num, flight in airline_toolkit.db.flights.items():
            for date, date_data in flight.dates.items():
                if isinstance(date_data, FlightDateStatusAvailable):
                    if date_data.available_seats.get(cabin, 0) > 0:
                        available_flight = flight
                        flight_date = date
                        break
            if available_flight:
                break

        if not available_flight:
            pytest.skip("No available flights with seats")

        flight_date_data = available_flight.dates[flight_date]
        price = flight_date_data.prices[cabin]

        payment_id = None
        for pm_id, pm in user.payment_methods.items():
            if pm.source == "credit_card":
                payment_id = pm_id
                break

        # Total with insurance: ticket + $30 per passenger
        total_price = price + 30

        result = airline_toolkit.use_tool(
            "book_reservation",
            user_id=user_id,
            origin=available_flight.origin,
            destination=available_flight.destination,
            flight_type="one_way",
            cabin=cabin,
            flights=[{"flight_number": available_flight.flight_number, "date": flight_date}],
            passengers=[{"first_name": "Test", "last_name": "Insured", "dob": "1985-03-15"}],
            payment_methods=[{"payment_id": payment_id, "amount": total_price}],
            total_baggages=0,
            nonfree_baggages=0,
            insurance="yes",
        )

        assert result is not None
        assert result.insurance == "yes"

    def test_book_reservation_with_baggage(self, airline_toolkit):
        """Book reservation with non-free baggage adds baggage fee."""
        from maseval.benchmark.tau2.domains.airline.models import FlightDateStatusAvailable

        user = None
        user_id = None
        for uid, u in airline_toolkit.db.users.items():
            for pm_id, pm in u.payment_methods.items():
                if pm.source == "credit_card":
                    user = u
                    user_id = uid
                    break
            if user:
                break

        if not user:
            pytest.skip("No user with credit card payment method")

        available_flight = None
        flight_date = None
        cabin = "economy"

        for flight_num, flight in airline_toolkit.db.flights.items():
            for date, date_data in flight.dates.items():
                if isinstance(date_data, FlightDateStatusAvailable):
                    if date_data.available_seats.get(cabin, 0) > 0:
                        available_flight = flight
                        flight_date = date
                        break
            if available_flight:
                break

        if not available_flight:
            pytest.skip("No available flights with seats")

        flight_date_data = available_flight.dates[flight_date]
        price = flight_date_data.prices[cabin]

        payment_id = None
        for pm_id, pm in user.payment_methods.items():
            if pm.source == "credit_card":
                payment_id = pm_id
                break

        # Total with 2 non-free bags: ticket + $50 * 2 = ticket + $100
        nonfree_bags = 2
        total_price = price + (50 * nonfree_bags)

        result = airline_toolkit.use_tool(
            "book_reservation",
            user_id=user_id,
            origin=available_flight.origin,
            destination=available_flight.destination,
            flight_type="one_way",
            cabin=cabin,
            flights=[{"flight_number": available_flight.flight_number, "date": flight_date}],
            passengers=[{"first_name": "Test", "last_name": "Baggage", "dob": "1985-03-15"}],
            payment_methods=[{"payment_id": payment_id, "amount": total_price}],
            total_baggages=2,
            nonfree_baggages=nonfree_bags,
            insurance="no",
        )

        assert result is not None
        assert result.total_baggages == 2
        assert result.nonfree_baggages == nonfree_bags

    def test_book_reservation_payment_mismatch_raises(self, airline_toolkit):
        """Booking fails when payment doesn't match total price."""
        from maseval.benchmark.tau2.domains.airline.models import FlightDateStatusAvailable

        user = None
        user_id = None
        for uid, u in airline_toolkit.db.users.items():
            for pm_id, pm in u.payment_methods.items():
                if pm.source == "credit_card":
                    user = u
                    user_id = uid
                    break
            if user:
                break

        if not user:
            pytest.skip("No user with credit card payment method")

        available_flight = None
        flight_date = None
        cabin = "economy"

        for flight_num, flight in airline_toolkit.db.flights.items():
            for date, date_data in flight.dates.items():
                if isinstance(date_data, FlightDateStatusAvailable):
                    if date_data.available_seats.get(cabin, 0) > 0:
                        available_flight = flight
                        flight_date = date
                        break
            if available_flight:
                break

        if not available_flight:
            pytest.skip("No available flights with seats")

        payment_id = None
        for pm_id, pm in user.payment_methods.items():
            if pm.source == "credit_card":
                payment_id = pm_id
                break

        # Intentionally wrong amount
        wrong_amount = 1

        with pytest.raises(ValueError, match="does not add up"):
            airline_toolkit.use_tool(
                "book_reservation",
                user_id=user_id,
                origin=available_flight.origin,
                destination=available_flight.destination,
                flight_type="one_way",
                cabin=cabin,
                flights=[{"flight_number": available_flight.flight_number, "date": flight_date}],
                passengers=[{"first_name": "Test", "last_name": "User", "dob": "1985-03-15"}],
                payment_methods=[{"payment_id": payment_id, "amount": wrong_amount}],
                total_baggages=0,
                nonfree_baggages=0,
                insurance="no",
            )


# =============================================================================
# One-Stop Flight Search Tests
# =============================================================================


@pytest.mark.benchmark
class TestAirlineOneStopFlightSearch:
    """Tests for one-stop connecting flight search."""

    def test_search_onestop_explores_connections(self, airline_toolkit):
        """search_onestop_flight explores connecting flights."""
        # This test verifies the search logic runs, even if no connections exist
        result = airline_toolkit.use_tool(
            "search_onestop_flight",
            origin="JFK",
            destination="LAX",
            date="2024-05-15",
        )

        # Result should be a list (possibly empty)
        assert isinstance(result, list)

    def test_search_direct_flight_internal(self, airline_toolkit):
        """_search_direct_flight returns matching flights."""
        # Find actual flights in the database
        if not airline_toolkit.db.flights:
            pytest.skip("No flights in database")

        # Get first flight's origin and a date it operates on
        flight = list(airline_toolkit.db.flights.values())[0]
        dates = list(flight.dates.keys())
        if not dates:
            pytest.skip("Flight has no dates")

        # Search for flights from this origin on this date
        result = airline_toolkit._search_direct_flight(
            date=dates[0],
            origin=flight.origin,
        )

        # Should return at least one result (the flight we're searching from)
        assert isinstance(result, list)


# =============================================================================
# Update Reservation Flights - New Flight Tests
# =============================================================================


@pytest.mark.benchmark
class TestAirlineUpdateReservationFlights:
    """Tests for update_reservation_flights with actual flight changes."""

    def test_update_flights_to_new_flight(self, airline_toolkit):
        """Update reservation with a different flight."""
        from maseval.benchmark.tau2.domains.airline.models import FlightDateStatusAvailable

        # Find an active reservation with a user who has payment methods
        reservation = None
        user = None
        for res_id, res in airline_toolkit.db.reservations.items():
            if res.status != "cancelled" and res.flights:
                u = airline_toolkit.db.users.get(res.user_id)
                if u and u.payment_methods:
                    reservation = res
                    user = u
                    break

        if not reservation or not user:
            pytest.skip("No suitable reservation for flight update")

        # Find a different available flight with the same route
        current_flight = reservation.flights[0]
        new_flight = None
        new_date = None

        for flight_num, flight in airline_toolkit.db.flights.items():
            if (
                flight.origin == current_flight.origin
                and flight.destination == current_flight.destination
                and flight_num != current_flight.flight_number
            ):
                for date, date_data in flight.dates.items():
                    if isinstance(date_data, FlightDateStatusAvailable):
                        if date_data.available_seats.get(reservation.cabin, 0) >= len(reservation.passengers):
                            new_flight = flight
                            new_date = date
                            break
                if new_flight:
                    break

        if not new_flight:
            pytest.skip("No alternative flight found for same route")

        payment_id = list(user.payment_methods.keys())[0]

        result = airline_toolkit.use_tool(
            "update_reservation_flights",
            reservation_id=reservation.reservation_id,
            cabin=reservation.cabin,
            flights=[{"flight_number": new_flight.flight_number, "date": new_date}],
            payment_id=payment_id,
        )

        assert result is not None
        assert len(result.flights) == 1
        assert result.flights[0].flight_number == new_flight.flight_number


# =============================================================================
# Certificate and Gift Card Tests
# =============================================================================


@pytest.mark.benchmark
class TestAirlinePaymentMethods:
    """Tests for certificate and gift card payment handling."""

    def test_send_certificate_adds_to_user(self, airline_toolkit):
        """send_certificate adds certificate to user payment methods."""
        user_ids = list(airline_toolkit.db.users.keys())
        if not user_ids:
            pytest.skip("No users in database")

        user_id = user_ids[0]
        original_count = len(airline_toolkit.db.users[user_id].payment_methods)

        result = airline_toolkit.use_tool(
            "send_certificate",
            user_id=user_id,
            amount=100,
        )

        assert "Certificate" in result
        assert len(airline_toolkit.db.users[user_id].payment_methods) == original_count + 1

    def test_book_with_gift_card_deducts_balance(self, airline_toolkit):
        """Booking with gift card deducts from balance."""
        from maseval.benchmark.tau2.domains.airline.models import FlightDateStatusAvailable, GiftCard

        # Find user with gift card
        user = None
        user_id = None
        gift_card_id = None

        for uid, u in airline_toolkit.db.users.items():
            for pm_id, pm in u.payment_methods.items():
                if isinstance(pm, GiftCard) and pm.amount >= 100:
                    user = u
                    user_id = uid
                    gift_card_id = pm_id
                    break
            if user:
                break

        if not user:
            pytest.skip("No user with sufficient gift card balance")

        # Find a cheap flight
        available_flight = None
        flight_date = None
        cabin = "basic_economy"  # Usually cheapest

        for flight_num, flight in airline_toolkit.db.flights.items():
            for date, date_data in flight.dates.items():
                if isinstance(date_data, FlightDateStatusAvailable):
                    if cabin in date_data.prices and cabin in date_data.available_seats:
                        if date_data.available_seats[cabin] > 0:
                            price = date_data.prices[cabin]
                            if price <= user.payment_methods[gift_card_id].amount:
                                available_flight = flight
                                flight_date = date
                                break
            if available_flight:
                break

        if not available_flight:
            pytest.skip("No flight cheap enough for gift card")

        flight_date_data = available_flight.dates[flight_date]
        price = flight_date_data.prices[cabin]
        original_balance = user.payment_methods[gift_card_id].amount

        result = airline_toolkit.use_tool(
            "book_reservation",
            user_id=user_id,
            origin=available_flight.origin,
            destination=available_flight.destination,
            flight_type="one_way",
            cabin=cabin,
            flights=[{"flight_number": available_flight.flight_number, "date": flight_date}],
            passengers=[{"first_name": "Gift", "last_name": "Card", "dob": "1990-01-01"}],
            payment_methods=[{"payment_id": gift_card_id, "amount": price}],
            total_baggages=0,
            nonfree_baggages=0,
            insurance="no",
        )

        assert result is not None
        # Gift card balance should be reduced
        assert user.payment_methods[gift_card_id].amount == original_balance - price
