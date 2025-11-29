"""Stock price tool implementation."""

from .base import BaseTool, ToolResult


class StockPriceTool(BaseTool):
    """Stock price lookup tool with static data.

    Provides stock price information for configured symbols.
    """

    def __init__(self, price_data: dict[str, float]):
        description = "Look up stock prices by symbol (e.g. 'AAPL', 'GOOGL')"
        super().__init__(
            "stock_price",
            description,
            tool_args=["action", "symbol"],
        )
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
