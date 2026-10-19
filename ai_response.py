"""
Backend Reply -> AI Response: routes each caller turn to one of two flows.

  * regenta_flow.py - city -> hotel -> questions about it -> transfer to the
    hotel's reservations / restaurant / banquets team (no booking taken).
  * support_agent.py - existing bookings: lookup, cancellation, tickets,
    hotel info, human transfer.

The first caller turn picks the flow (session.flow). After that the call can
switch: support -> regenta when its model calls startNewBooking, and
regenta -> support when the caller asks about an existing booking.

Before either flow, call_language.py asks which language to use and keeps the
call in it: the flows read and write English, and call_language translates the
caller's words in and each reply out.

generate_reply() is still the conversation loop's single entry point.
"""
from __future__ import annotations

import re

import booking_agent
import call_language
import regenta_flow
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

def _strip_markers(text: str) -> str:
    return re.sub(r"^(\s*\[[^\]]*\]\s*)+", "", text).strip()


def _first_flow(text: str) -> str:
    if _EXISTING_RE.search(text):
        return "support"
    return "regenta"   # the greeting asked for a city


def _drop_last_user_message(session: Session) -> None:
    if session.conversation and session.conversation[-1]["role"] == "user":
        session.conversation.pop()


async def _to_booking(session: Session, user_message: str) -> str:
    session.flow = "regenta"
    session.wants_new_booking = False
    session.awaiting_transfer = False
    print("  [router] -> regenta flow")
    return await regenta_flow.generate_reply(session, user_message)


async def generate_reply(session: Session, user_message: str) -> str:
    """Language choice / switch first; then the flow in English, reply translated back."""
    if session.call_language is None:
        return await call_language.choose(session, user_message, _route)
    english = await call_language.to_english(session, user_message)
    switched = await call_language.switch_request(session, user_message, english)
    if switched is not None:
        return switched
    return await call_language.localize(session, await _route(session, english))


async def _route(session: Session, user_message: str) -> str:
    text = _strip_markers(user_message)

    if session.flow is None:
        session.flow = _first_flow(text)
        print(f"  [router] -> {session.flow} flow")
    elif session.flow == "regenta" and _EXISTING_RE.search(text):
        session.flow = "support"
        print("  [router] -> support flow")

    if session.flow == "regenta":
        return await regenta_flow.generate_reply(session, user_message)

    reply = await support_agent.generate_reply(session, user_message)
    if session.wants_new_booking and session.stage != Stage.TRANSFERRED:
        _drop_last_user_message(session)   # booking flow records the message itself
        return await _to_booking(session, user_message)
    return reply


def opening_lines(session: Session) -> list[tuple[str, str]]:
    """[(text, language)]: the greeting in English, then in Hindi (languages.json)."""
    greeting = call_language.LANGUAGES["greeting"]
    session.add_message("assistant", greeting[0]["text"])
    return [(g["text"], g["lang"]) for g in greeting]


def opening_line(session: Session) -> str:
    """The greeting as one string (main.py, the local mic/file test)."""
    return " ".join(text for text, _ in opening_lines(session))


async def close() -> None:
    await booking_agent.close()
    await support_agent.close()
    await call_language.close()
