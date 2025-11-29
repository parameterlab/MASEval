"""Calendar tool implementation."""

from dataclasses import dataclass

from .base import BaseTool, ToolResult


@dataclass
class TimeSlot:
    """Time slot representation."""

    start: str
    end: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {"start": self.start, "end": self.end}


class CalendarTool(BaseTool):
    """Calendar tool with availability and event management."""

    def __init__(self, calendar_id: str, availability_data: dict[str, list[dict[str, str]]]):
        super().__init__(
            f"calendar_{calendar_id}",
            f"Access {calendar_id} calendar",
            tool_args=["action", "start_date", "end_date", "date"],
        )
        self.calendar_id = calendar_id
        self.availability: dict[str, list[TimeSlot]] = {}

        # Load availability
        for date, slots in availability_data.items():
            self.availability[date] = [TimeSlot(start=s["start"], end=s["end"]) for s in slots]

    def execute(self, **kwargs) -> ToolResult:
        """Execute calendar action."""
        action = kwargs.get("action")

        if action == "get_availability":
            return self._get_availability(
                start_date=kwargs.get("start_date"),
                end_date=kwargs.get("end_date"),
            )
        elif action == "get_date_availability":
            return self._get_date_availability(kwargs.get("date"))
        else:
            return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

    def _get_availability(self, start_date: str | None = None, end_date: str | None = None) -> ToolResult:
        """Get availability for date range."""
        if not start_date and not end_date:
            # Return all availability
            result = {date: [slot.to_dict() for slot in slots] for date, slots in self.availability.items()}
        else:
            # Filter by date range
            result = {}
            for date, slots in self.availability.items():
                if start_date and date < start_date:
                    continue
                if end_date and date > end_date:
                    continue
                result[date] = [slot.to_dict() for slot in slots]

        return ToolResult(
            success=True,
            data={"availability": result, "calendar_id": self.calendar_id},
        )

    def _get_date_availability(self, date: str | None) -> ToolResult:
        """Get availability for a specific date."""
        if not date:
            return ToolResult(success=False, data=None, error="date is required")

        if date in self.availability:
            slots = [slot.to_dict() for slot in self.availability[date]]
            return ToolResult(
                success=True,
                data={
                    "date": date,
                    "slots": slots,
                    "calendar_id": self.calendar_id,
                },
            )

        return ToolResult(
            success=True,
            data={"date": date, "slots": [], "calendar_id": self.calendar_id},
            metadata={"message": "No availability for this date"},
        )
