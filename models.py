"""
Plain dataclasses for the support workflow (SupportStage, SupportTicket).

No MongoDB coupling: mongo.py is the only place that turns these into stored
documents. This file must NOT import session.py (session.py imports from here).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

class SupportStage(str, Enum):
    """Support workflow stage; separate from Stage in session.py, which only
    says whether the call is still in progress."""

    NONE = "none"                          # no support issue raised yet
    CHECKING_CONTEXT = "checking_context"  # booking verified, looking into the issue
    RESOLVING = "resolving"                # attempting resolution
    TICKETED = "ticketed"                  # createSupportTicket succeeded
    ESCALATED = "escalated"                # transferToHuman fired
    RESOLVED = "resolved"                  # confirmed fixed with customer


@dataclass
class SupportTicket:
    """§9/§5 — created by createSupportTicket when an issue can't be
    resolved on the call. Kept as a plain dataclass on Session, same
    as booking, so mongo.py's save_support_ticket() can serialize it
    without this file needing to know about Mongo."""

    ticket_id: str | None = None
    category: str | None = None
    description: str | None = None
    priority: str = "normal"            # low / normal / high / urgent
    status: str = "open"                # open / in_progress / pending_customer /
                                         # pending_approval / resolved / closed
    booking_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "category": self.category,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "booking_id": self.booking_id,
        }