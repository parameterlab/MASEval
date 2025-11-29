"""Family information tool."""

from typing import Any

from .base import BaseTool, ToolResult


class FamilyInfoTool(BaseTool):
    """Family information lookup tool.
    
    Provides access to family member information and asset ownership data.
    """

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
