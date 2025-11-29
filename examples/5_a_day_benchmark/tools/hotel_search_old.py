"""Hotel search tool for optimization task."""

from typing import Any

from .base import BaseTool, ToolResult


class HotelSearchTool(BaseTool):
    """Hotel search and retrieval tool."""

    def __init__(self, hotels_data: list[dict[str, Any]]):
        description = (
            "Search and retrieve hotel information. "
            "Actions: 'search' (filter hotels by max_price, max_distance, min_wifi), "
            "'get_hotel' (get hotel by hotel_id), 'get_all' (get all hotels)"
        )
        super().__init__(
            "hotel_search",
            description,
            tool_args=["action", "max_price", "max_distance", "min_wifi", "hotel_id"],
        )
        self.hotels = hotels_data

    def execute(self, **kwargs) -> ToolResult:
        """Execute hotel search action."""
        action = kwargs.get("action", "search")

        if action == "search":
            return self._search_hotels(
                max_price=kwargs.get("max_price"),
                max_distance=kwargs.get("max_distance"),
                min_wifi=kwargs.get("min_wifi"),
            )
        elif action == "get_hotel":
            return self._get_hotel(kwargs.get("hotel_id"))
        elif action == "get_all":
            return self._get_all_hotels()
        else:
            return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

    def _search_hotels(
        self,
        max_price: float | None = None,
        max_distance: float | None = None,
        min_wifi: int | None = None,
    ) -> ToolResult:
        """Search hotels with filters."""
        filtered = self.hotels

        if max_price is not None:
            filtered = [h for h in filtered if h["price_per_night"] <= max_price]
        if max_distance is not None:
            filtered = [h for h in filtered if h["distance_to_venue_km"] <= max_distance]
        if min_wifi is not None:
            filtered = [h for h in filtered if h["wifi_speed_mbps"] >= min_wifi]

        return ToolResult(
            success=True,
            data={"hotels": filtered, "count": len(filtered)},
        )

    def _get_hotel(self, hotel_id: str | None) -> ToolResult:
        """Get specific hotel by ID."""
        if not hotel_id:
            return ToolResult(success=False, data=None, error="hotel_id is required")

        for hotel in self.hotels:
            if hotel["id"] == hotel_id:
                return ToolResult(success=True, data=hotel)

        return ToolResult(success=False, data=None, error=f"Hotel {hotel_id} not found")

    def _get_all_hotels(self) -> ToolResult:
        """Get all available hotels."""
        return ToolResult(
            success=True,
            data={"hotels": self.hotels, "count": len(self.hotels)},
        )
