"""Email tool collection with shared state across sub-tools.

Tools:
- email_get_inbox: List all emails in inbox
- email_get: Read specific email by ID
- email_send: Send an email
- email_draft: Draft an email without sending
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .base import BaseTool, ToolResult


@dataclass
class EmailMessage:
    """Email message structure."""

    id: str
    from_address: str
    to_address: str | None
    subject: str
    body: str
    timestamp: str
    read: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "from": self.from_address,
            "to": self.to_address,
            "subject": self.subject,
            "body": self.body,
            "timestamp": self.timestamp,
            "read": self.read,
        }


class EmailState:
    """Shared state for all email tools.

    Maintains inbox and sent emails across all sub-tool invocations.
    """

    def __init__(self, inbox_data: list[dict[str, Any]]):
        self.inbox: list[EmailMessage] = []
        self.sent: list[EmailMessage] = []
        self._next_id = len(inbox_data) + 1

        # Load inbox
        for i, email_data in enumerate(inbox_data):
            self.inbox.append(
                EmailMessage(
                    id=str(i + 1),
                    from_address=email_data["from"],
                    to_address=None,
                    subject=email_data["subject"],
                    body=email_data["body"],
                    timestamp=email_data["timestamp"],
                    read=False,
                )
            )


class EmailGetInboxTool(BaseTool):
    """Get all emails in inbox."""

    def __init__(self, email_state: EmailState):
        super().__init__(
            "email_get_inbox",
            "List all emails in the inbox",
            tool_args=[],
        )
        self.state = email_state

    def execute(self, **kwargs) -> ToolResult:
        """Get all emails in inbox."""
        emails = [email.to_dict() for email in self.state.inbox]
        return ToolResult(
            success=True,
            data=emails,
            metadata={"count": len(emails)},
        )


class EmailGetTool(BaseTool):
    """Read a specific email by ID."""

    def __init__(self, email_state: EmailState):
        super().__init__(
            "email_get",
            "Read a specific email by ID",
            tool_args=["email_id"],
        )
        self.state = email_state

    def execute(self, **kwargs) -> ToolResult:
        """Get specific email by ID."""
        email_id = kwargs.get("email_id")

        if not email_id:
            return ToolResult(success=False, data=None, error="email_id is required")

        for email in self.state.inbox:
            if email.id == email_id:
                email.read = True
                return ToolResult(success=True, data=email.to_dict())

        return ToolResult(success=False, data=None, error=f"Email {email_id} not found")


class EmailSendTool(BaseTool):
    """Send an email."""

    def __init__(self, email_state: EmailState):
        super().__init__(
            "email_send",
            "Send an email to a recipient",
            tool_args=["to", "subject", "body"],
        )
        self.state = email_state

    def execute(self, **kwargs) -> ToolResult:
        """Send an email."""
        to = kwargs.get("to")
        subject = kwargs.get("subject")
        body = kwargs.get("body")

        if not to or not subject or not body:
            return ToolResult(
                success=False,
                data=None,
                error="to, subject, and body are required",
            )

        email = EmailMessage(
            id=str(self.state._next_id),
            from_address="me@example.com",
            to_address=to,
            subject=subject,
            body=body,
            timestamp=datetime.now().isoformat(),
            read=True,
        )
        self.state._next_id += 1
        self.state.sent.append(email)

        return ToolResult(
            success=True,
            data={"status": "sent", "id": email.id, "to": to},
        )


class EmailDraftTool(BaseTool):
    """Draft an email without sending."""

    def __init__(self, email_state: EmailState):
        super().__init__(
            "email_draft",
            "Draft an email without sending it",
            tool_args=["to", "subject", "body"],
        )
        self.state = email_state

    def execute(self, **kwargs) -> ToolResult:
        """Draft an email without sending."""
        to = kwargs.get("to")
        subject = kwargs.get("subject")
        body = kwargs.get("body")

        if not to or not subject or not body:
            return ToolResult(
                success=False,
                data=None,
                error="to, subject, and body are required",
            )

        draft = {
            "to": to,
            "subject": subject,
            "body": body,
            "status": "draft",
        }

        return ToolResult(success=True, data=draft)


class EmailToolCollection:
    """Email tool collection factory.

    Creates a shared state and returns all email sub-tools that share that state.
    This ensures sent emails are visible across all email operations.

    Usage:
        collection = EmailToolCollection(inbox_data)
        tools = collection.get_sub_tools()
        # All tools share the same inbox and sent folder
    """

    def __init__(self, inbox_data: list[dict[str, Any]]):
        self.state = EmailState(inbox_data)

    def get_sub_tools(self) -> list[BaseTool]:
        """Return all email sub-tools with shared state."""
        return [
            EmailGetInboxTool(self.state),
            EmailGetTool(self.state),
            EmailSendTool(self.state),
            EmailDraftTool(self.state),
        ]
