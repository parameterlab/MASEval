"""Banking tool implementation."""

from dataclasses import dataclass
from typing import Any

from .base import BaseTool, ToolResult


@dataclass
class Transaction:
    """Bank transaction structure."""

    id: str
    date: str
    description: str
    amount: float
    type: str  # "deposit" or "expense"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "date": self.date,
            "description": self.description,
            "amount": self.amount,
            "type": self.type,
        }


class BankingTool(BaseTool):
    """Banking tool with transaction history, balance, and asset information."""

    def __init__(
        self,
        transactions_data: list[dict[str, Any]],
        current_balance: float,
        assets_data: dict[str, Any] | None = None,
    ):
        description = (
            "Access bank account information and assets. "
            "Actions: 'get_balance' (get current balance), 'get_transactions' (get transactions with optional start_date/end_date), "
            "'get_transaction' (get transaction by transaction_id), "
            "'get_asset' (requires asset_name parameter, e.g. asset_name='AAPL')"
        )
        super().__init__(
            "banking",
            description,
            tool_args=["action", "transaction_id", "start_date", "end_date", "asset_name"],
        )
        self.current_balance = current_balance
        self.transactions: list[Transaction] = []

        # Store assets
        self.assets = assets_data or {}

        # Load transactions
        for i, txn_data in enumerate(transactions_data):
            self.transactions.append(
                Transaction(
                    id=str(i + 1),
                    date=txn_data["date"],
                    description=txn_data["description"],
                    amount=txn_data["amount"],
                    type=txn_data["type"],
                )
            )

    def execute(self, **kwargs) -> ToolResult:
        """Execute banking action."""
        action = kwargs.get("action")

        if action == "get_balance":
            return self._get_balance()
        elif action == "get_transactions":
            return self._get_transactions(
                start_date=kwargs.get("start_date"),
                end_date=kwargs.get("end_date"),
            )
        elif action == "get_transaction":
            return self._get_transaction(kwargs.get("transaction_id"))
        elif action == "get_asset":
            return self._get_asset(kwargs.get("asset_name"))
        else:
            return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

    def _get_balance(self) -> ToolResult:
        """Get current account balance."""
        return ToolResult(
            success=True,
            data={"balance": self.current_balance, "currency": "USD"},
        )

    def _get_transactions(self, start_date: str | None = None, end_date: str | None = None) -> ToolResult:
        """Get transaction history with optional date filtering."""
        filtered = self.transactions

        if start_date:
            filtered = [t for t in filtered if t.date >= start_date]
        if end_date:
            filtered = [t for t in filtered if t.date <= end_date]

        return ToolResult(
            success=True,
            data=[t.to_dict() for t in filtered],
            metadata={"count": len(filtered)},
        )

    def _get_transaction(self, transaction_id: str | None) -> ToolResult:
        """Get specific transaction by ID."""
        if not transaction_id:
            return ToolResult(success=False, data=None, error="transaction_id is required")

        for txn in self.transactions:
            if txn.id == transaction_id:
                return ToolResult(success=True, data=txn.to_dict())

        return ToolResult(success=False, data=None, error=f"Transaction {transaction_id} not found")

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
