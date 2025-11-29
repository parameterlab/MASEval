"""Tools for the 5-a-day benchmark."""

from .base import BaseTool, ToolResult, ToolInvocation
from .email import EmailToolCollection
from .banking import BankingToolCollection
from .calculator import CalculatorToolCollection
from .code_execution import CodeExecutionToolCollection
from .family_info import FamilyInfoToolCollection
from .stock_price import StockPriceToolCollection
from .calendar import CalendarToolCollection
from .fitness import RunningAppToolCollection, GymTrackerToolCollection
from .hotel_search import HotelSearchToolCollection
from .factory import ToolFactory
from .mcp_calendar import MCPCalendarToolCollection

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolInvocation",
    "EmailToolCollection",
    "BankingToolCollection",
    "CalculatorToolCollection",
    "CodeExecutionToolCollection",
    "FamilyInfoToolCollection",
    "StockPriceToolCollection",
    "CalendarToolCollection",
    "RunningAppToolCollection",
    "GymTrackerToolCollection",
    "HotelSearchToolCollection",
    "ToolFactory",
    "MCPCalendarToolCollection",
]
