"""MCP Calendar Tool - Model Context Protocol Integration

Demonstrates MCP integration by wrapping calendar operations as framework-agnostic tools.
Shows how maseval enables agents to work with external systems via standardized protocols.
"""

import json
from typing import Any, Dict

from .base import BaseTool, ToolResult


class MCPCalendarTool(BaseTool):
    """Calendar tool demonstrating MCP protocol integration."""

    def __init__(self, name: str, calendar_data: Dict[str, Any]):
        """Initialize MCP calendar tool.

        Args:
            name: Tool name (e.g., "my_calendar_mcp", "other_calendar_mcp")
            calendar_data: Calendar events dict with "events" list
        """
        description = f"Access {name} via MCP. Supports: get_events, check_availability, add_event"
        super().__init__(name, description)
        self.calendar_data = calendar_data

    def execute(self, **kwargs) -> ToolResult:
        """Execute calendar operation via MCP protocol pattern."""
        action = kwargs.get("action")
        if not action:
            return ToolResult(success=False, data=None, error="Missing 'action' parameter")

        try:
            if action == "get_events":
                result = self._mcp_get_events(
                    kwargs.get("start_date", ""), kwargs.get("end_date", "")
                )
            elif action == "check_availability":
                result = self._mcp_check_availability(
                    kwargs.get("date", ""),
                    kwargs.get("start_time", ""),
                    kwargs.get("end_time", ""),
                )
            elif action == "add_event":
                result = self._mcp_add_event(
                    kwargs.get("date", ""),
                    kwargs.get("start_time", ""),
                    kwargs.get("end_time", ""),
                    kwargs.get("title", ""),
                    kwargs.get("description", ""),
                )
            else:
                return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

            return ToolResult(success=True, data=result, metadata={"mcp": True, "action": action})

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def _mcp_get_events(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get events in date range."""
        events = self.calendar_data.get("events", [])
        if start_date and end_date:
            filtered = [e for e in events if start_date <= e["date"] <= end_date]
        elif start_date:
            filtered = [e for e in events if e["date"] >= start_date]
        elif end_date:
            filtered = [e for e in events if e["date"] <= end_date]
        else:
            filtered = events
        return {"events": filtered, "count": len(filtered)}

    def _mcp_check_availability(self, date: str, start_time: str, end_time: str) -> Dict[str, Any]:
        """Check if time slot is available."""
        events = self.calendar_data.get("events", [])
        conflicts = [
            e
            for e in events
            if e["date"] == date
            and not (end_time <= e["start_time"] or start_time >= e["end_time"])
        ]
        return {"available": len(conflicts) == 0, "conflicts": conflicts}

    def _mcp_add_event(
        self, date: str, start_time: str, end_time: str, title: str, description: str = ""
    ) -> Dict[str, Any]:
        """Add a new event to calendar."""
        if not all([date, start_time, end_time, title]):
            raise ValueError("Missing required fields: date, start_time, end_time, title")

        event = {
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "title": title,
            "description": description,
        }
        self.calendar_data.setdefault("events", []).append(event)
        return {"success": True, "event": event}
