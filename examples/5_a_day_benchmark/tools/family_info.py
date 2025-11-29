"""Family information tool."""

from typing import Any

from .base import BaseTool, ToolResult


class FamilyInfoTool(BaseTool):
    """Family information lookup tool.

    Provides access to family member information and asset ownership data.
    """

    def __init__(self, family_data: dict[str, Any]):
        description = "Access family information. Actions: 'get_children' (get list of children)"
        super().__init__(
            "family_info",
            description,
            tool_args=["action"],
        )
        self.children = family_data.get("children", [])

    def execute(self, **kwargs) -> ToolResult:
        """Execute family info action."""
        action = kwargs.get("action")

        if action == "get_children":
            return self._get_children()
        else:
            return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

    def _get_children(self) -> ToolResult:
        """Get list of children."""
        return ToolResult(
            success=True,
            data={"children": self.children, "count": len(self.children)},
        )
