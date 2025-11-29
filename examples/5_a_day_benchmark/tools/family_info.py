"""Family info and stock price tools."""

from typing import Any

from .base import BaseTool, ToolResult


class FamilyInfoTool(BaseTool):
    """Family information lookup tool."""

    def __init__(self, family_data: dict[str, Any]):
        description = (
            "Access family information and assets. "
            "Actions: 'get_children' (get list of children), "
            "'get_asset' (requires asset_name parameter, e.g. asset_name='apple_shares')"
        )
        super().__init__("family_info", description)
        self.children = family_data.get("children", [])
        self.assets = {
            "apple_shares": family_data.get("apple_shares_owned", 0),
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute family info action."""
        action = kwargs.get("action")

        if action == "get_children":
            return self._get_children()
        elif action == "get_asset":
            return self._get_asset(kwargs.get("asset_name"))
        else:
            return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

    def _get_children(self) -> ToolResult:
        """Get list of children."""
        return ToolResult(
            success=True,
            data={"children": self.children, "count": len(self.children)},
        )

    def _get_asset(self, asset_name: str | None) -> ToolResult:
        """Get asset information."""
        if not asset_name:
            return ToolResult(success=False, data=None, error="asset_name is required")

        if asset_name in self.assets:
            return ToolResult(
                success=True,
                data={"asset": asset_name, "value": self.assets[asset_name]},
            )

        return ToolResult(success=False, data=None, error=f"Asset {asset_name} not found")


class StockPriceTool(BaseTool):
    """Stock price lookup tool with static data."""

    def __init__(self, price_data: dict[str, float]):
        description = (
            "Look up stock prices. "
            "Actions: 'get_price' (get stock price by symbol like 'AAPL', 'GOOGL')"
        )
        super().__init__("stock_price", description)
        self.prices = price_data

    def execute(self, **kwargs) -> ToolResult:
        """Execute stock price lookup."""
        action = kwargs.get("action", "get_price")

        if action == "get_price":
            return self._get_price(kwargs.get("symbol"))
        else:
            return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

    def _get_price(self, symbol: str | None) -> ToolResult:
        """Get stock price for symbol."""
        if not symbol:
            return ToolResult(success=False, data=None, error="symbol is required")

        symbol = symbol.upper()
        if symbol in self.prices:
            return ToolResult(
                success=True,
                data={
                    "symbol": symbol,
                    "price": self.prices[symbol],
                    "currency": "USD",
                },
            )

        return ToolResult(success=False, data=None, error=f"Symbol {symbol} not found")
