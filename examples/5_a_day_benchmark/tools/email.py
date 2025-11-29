"""Email tool implementation."""

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


class EmailTool(BaseTool):
    """Email tool with inbox and sending capabilities."""

    def __init__(self, inbox_data: list[dict[str, Any]]):
        description = (
            "Access and send emails. "
            "Actions: 'get_inbox' (lists all emails), 'get_email' (read email by email_id), "
            "'send_email' (send email with to, subject, body), 'draft_email' (draft email with to, subject, body)"
        )
        super().__init__("email", description)
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

    def execute(self, **kwargs) -> ToolResult:
        """Execute email action."""
        action = kwargs.get("action")
        if action == "get_inbox":
            return self._get_inbox()
        elif action == "get_email":
            return self._get_email(kwargs.get("email_id"))
        elif action == "send_email":
            return self._send_email(
                to=kwargs.get("to"),
                subject=kwargs.get("subject"),
                body=kwargs.get("body"),
            )
        elif action == "draft_email":
            return self._draft_email(
                to=kwargs.get("to"),
                subject=kwargs.get("subject"),
                body=kwargs.get("body"),
            )
        else:
            return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

    def _get_inbox(self) -> ToolResult:
        """Get all emails in inbox."""
        emails = [email.to_dict() for email in self.inbox]
        return ToolResult(
            success=True,
            data=emails,
            metadata={"count": len(emails)},
        )

    def _get_email(self, email_id: str | None) -> ToolResult:
        """Get specific email by ID."""
        if not email_id:
            return ToolResult(success=False, data=None, error="email_id is required")

        for email in self.inbox:
            if email.id == email_id:
                email.read = True
                return ToolResult(success=True, data=email.to_dict())

        return ToolResult(success=False, data=None, error=f"Email {email_id} not found")

    def _send_email(self, to: str | None, subject: str | None, body: str | None) -> ToolResult:
        """Send an email."""
        if not to or not subject or not body:
            return ToolResult(
                success=False,
                data=None,
                error="to, subject, and body are required",
            )

        email = EmailMessage(
            id=str(self._next_id),
            from_address="me@example.com",
            to_address=to,
            subject=subject,
            body=body,
            timestamp=datetime.now().isoformat(),
            read=True,
        )
        self._next_id += 1
        self.sent.append(email)

        return ToolResult(
            success=True,
            data={"status": "sent", "id": email.id, "to": to},
        )

    def _draft_email(self, to: str | None, subject: str | None, body: str | None) -> ToolResult:
        """Draft an email without sending."""
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
