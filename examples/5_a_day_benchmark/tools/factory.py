"""Tool factory for creating tools from task data."""

from typing import Any

from .banking import BankingTool
from .calculator import CalculatorTool
from .calendar import CalendarTool
from .code_execution import CodeExecutionTool
from .email import EmailTool
from .family_info import FamilyInfoTool
from .fitness import GymTrackerTool, RunningAppTool
from .hotel_search import HotelSearchTool
from .stock_price import StockPriceTool


class ToolFactory:
    """Factory for creating tools from task environment data."""

    @staticmethod
    def create_tools(environment_data: dict[str, Any]) -> dict[str, Any]:
        """Create tools based on task environment_data.

        Args:
            environment_data: Task environment configuration

        Returns:
            Dictionary mapping tool names to tool instances
        """
        tools = {}
        required_tools = environment_data.get("tools", [])

        # Email tool
        if "email" in required_tools:
            inbox_data = environment_data.get("email_inbox", [])
            tools["email"] = EmailTool(inbox_data=inbox_data)

        # Banking tool
        if "banking" in required_tools:
            transactions_data = environment_data.get("bank_transactions", [])
            current_balance = environment_data.get("current_balance", 0.0)
            tools["banking"] = BankingTool(
                transactions_data=transactions_data,
                current_balance=current_balance,
            )

        # Calculator tool
        if "calculator" in required_tools:
            tools["calculator"] = CalculatorTool()

        # Code execution tool
        if "python_executor" in required_tools:
            test_cases = environment_data.get("test_cases", [])
            tools["python_executor"] = CodeExecutionTool(test_cases=test_cases)

        # Family info tool
        if "family_info" in required_tools:
            family_data = {
                "children": environment_data.get("family_info", {}).get("children", []),
                "apple_shares_owned": environment_data.get("family_info", {}).get("apple_shares_owned", 0),
            }
            tools["family_info"] = FamilyInfoTool(family_data=family_data)

        # Stock price tool
        if "stock_price" in required_tools or "websearch" in required_tools:
            # For websearch, we use stock price for deterministic results
            price_data = environment_data.get("stock_price_lookup", {})
            tools["stock_price"] = StockPriceTool(price_data=price_data)

        # Calendar tools (can have multiple calendars)
        if "my_calendar_mcp" in required_tools:
            my_availability = environment_data.get("my_calendar_availability", {})
            tools["my_calendar_mcp"] = CalendarTool(
                calendar_id="my_calendar",
                availability_data=my_availability,
            )

        if "other_calendar_mcp" in required_tools:
            other_person = environment_data.get("other_person_name", "other_person")
            other_availability = environment_data.get(f"{other_person}_availability", {})
            tools["other_calendar_mcp"] = CalendarTool(
                calendar_id=other_person,
                availability_data=other_availability,
            )

        # Running app tool
        if "running_app" in required_tools:
            running_activities = environment_data.get("running_activities", [])
            tools["running_app"] = RunningAppTool(activities_data=running_activities)

        # Gym tracker tool
        if "gym_tracker" in required_tools:
            gym_activities = environment_data.get("gym_activities", [])
            tools["gym_tracker"] = GymTrackerTool(workouts_data=gym_activities)

        # Hotel search tool
        if "hotel_search" in required_tools:
            hotels_data = environment_data.get("hotels", [])
            tools["hotel_search"] = HotelSearchTool(hotels_data=hotels_data)

        return tools
