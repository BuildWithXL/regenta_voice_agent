"""
Backend Reply -> AI Response: routes each caller turn to one of two flows.

  * booking_agent.py - new bookings (city -> hotel -> guest details -> confirm)
    plus hotel questions while booking.
  * support_agent.py - existing bookings: lookup, cancellation, tickets,
    hotel info, human transfer.

The first caller turn picks the flow (session.flow). After that the call can
switch: support -> booking when its model calls startNewBooking, and
booking -> support when the caller asks about an existing booking while no
new booking is half-filled.

generate_reply() is still the conversation loop's single entry point.
"""
from __future__ import annotations

import re

import booking_agent
import support_agent
from session import Session, Stage
from support_agent import build_handoff_summary  # noqa: F401  (used by main.py)

# "my booking", "cancel", a spoken booking ID, tickets/refunds: support.
_EXISTING_RE = re.compile(
    r"\b(my|existing|current|previous|old)\s+(booking|reservation)s?\b"
    r"|\b(have|had|made)\s+(a|an)\s+(booking|reservation)\b|\balready\s+booked\b"
    r"|\bcancel|\bbooking\s*(id|number)\b|\bbk[\s-]*[a-z0-9]{4,}"
    r"|\bticket\b|\brefund|\bcomplain|\bmodif|\bchange\s+(my|the)\s+(date|room|booking)",
    re.I,
)
# Asking to book something new: booking.
_NEW_RE = re.compile(
    r"\b(book|reserve)\b|\bnew\s+(booking|reservation)\b|\b(make|do)\s+a\s+(booking|reservation)\b"
    r"|\brooms?\s+(available|availability)\b|\bavailab\w*\s+rooms?\b"
    r"|बुक|बुकिंग\s*कर|booking\s+kar|book\s+kar",
    re.I,
)


def _strip_markers(text: str) -> str:
    return re.sub(r"^(\s*\[[^\]]*\]\s*)+", "", text).strip()


def _booking_in_progress(session: Session) -> bool:
    """True once the caller has started filling a new booking, so a stray
    word like 'change' doesn't throw away what they already gave."""
    if session.stage in (Stage.AWAITING_CONFIRMATION, Stage.BOOKING):
        return True
    if session.stage in (Stage.POST_BOOKING, Stage.INFO):
        return False
    return any(v is not None for v in session.checklist.to_dict().values())


def _first_flow(text: str) -> str:
    if _EXISTING_RE.search(text):
        return "support"
    if _NEW_RE.search(text):
        return "booking"
    return "support"   # small talk / hotel info; it hands off via startNewBooking


def _drop_last_user_message(session: Session) -> None:
    if session.conversation and session.conversation[-1]["role"] == "user":
        session.conversation.pop()


async def _to_booking(session: Session, user_message: str) -> str:
    session.flow = "booking"
    session.wants_new_booking = False
    session.awaiting_transfer = False
    if session.booking is not None or session.stage in (Stage.POST_BOOKING, Stage.INFO):
        session.start_new_booking()
    print("  [router] -> booking flow")
    return await booking_agent.generate_reply(session, user_message)


async def generate_reply(session: Session, user_message: str) -> str:
    text = _strip_markers(user_message)

    if session.flow is None:
        session.flow = _first_flow(text)
        print(f"  [router] -> {session.flow} flow")
    elif (session.flow == "booking" and _EXISTING_RE.search(text)
          and not _booking_in_progress(session)):
        session.flow = "support"
        print("  [router] -> support flow")

    if session.flow == "booking":
        return await booking_agent.generate_reply(session, user_message)

    reply = await support_agent.generate_reply(session, user_message)
    if session.wants_new_booking and session.stage != Stage.TRANSFERRED:
        _drop_last_user_message(session)   # booking flow records the message itself
        return await _to_booking(session, user_message)
    return reply


def opening_line(session: Session) -> str:
    return booking_agent.opening_line(session)


async def close() -> None:
    await booking_agent.close()
    await support_agent.close()
