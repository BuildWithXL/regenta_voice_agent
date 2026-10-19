"""Booking tools for support calls: look up (and verify) a booking, cancel it."""
from __future__ import annotations

import re
from typing import Any

import mongo
from mongo import _norm_booking_id as norm_booking_id

_ID_RE = re.compile(r"BK-[A-Z0-9]{8}")


def _norm_email(e: str) -> str:
    e = re.sub(r"\s+", "", e or "").lower()
    local, _, dom = e.partition("@")
    if dom in ("gmail.com", "googlemail.com"):
        local = local.split("+")[0].replace(".", "")
    return f"{local}@{dom}"


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "getBookingDetails",
            "description": (
                "Look up an existing booking and verify the caller. Needs the "
                "booking ID and the email used for the booking. Returns the "
                "booking details only if the email matches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string"},
                    "email": {"type": "string", "description": "lowercase"},
                },
                "required": ["booking_id", "email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancelBooking",
            "description": (
                "Cancel a booking that was already verified with getBookingDetails. "
                "Set confirmed=true ONLY after reading the booking back and the "
                "caller explicitly said yes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string"},
                    "confirmed": {"type": "boolean"},
                },
                "required": ["booking_id", "confirmed"],
            },
        },
    },
]


async def getBookingDetails(booking_id: str, email: str, **_: Any) -> dict[str, Any]:
    bid = norm_booking_id(booking_id or "")
    if not _ID_RE.fullmatch(bid):
        return {"found": False, "verified": False, "reason": "invalid_booking_id_format",
                "next_step": "Ask the caller to repeat the booking ID."}
    doc = await mongo.find_booking_for_caller(bid, email)
    print(f"[lookup] id={bid} found={bool(doc)}")  # no email/PII in logs

    # Same answer for "no such booking" and "wrong email" so IDs can't be probed.
    if not doc:
        return {"found": False, "verified": False, "reason": "no_match",
                   "next_step": "Say you could not match them. Ask the caller to say the booking ID again, then the email again, one at a time, and read each back before retrying."}

    c = doc.get("checklist") or {}
    b = doc.get("booking") or {}
    rooms = b.get("rooms") or doc.get("rooms")
    name = (doc.get("name")
            or " ".join(filter(None, [c.get("first_name"), c.get("last_name")])) or None)
    stored = next(iter(mongo._stored_ids(doc)), bid)
    return {
        "found": True,
        "verified": True,
        "booking_id": norm_booking_id(stored),
        "guest_name": name,
        "hotel": doc.get("hotel") or c.get("hotel"),
        "city": c.get("city"),
        "check_in": doc.get("checkIn") or c.get("check_in"),
        "check_out": doc.get("checkOut") or c.get("check_out"),
        "guests": doc.get("guests"),
        "adults": c.get("adults"),
        "children": c.get("children"),
        "room_type": doc.get("roomType"),
        "rooms": rooms,
        "nights": doc.get("nights"),
        "total_per_night": b.get("total_per_night") or doc.get("pricePerNight"),
        "total_amount": doc.get("totalAmount"),
        "currency": b.get("currency"),
        "status": doc.get("status", "confirmed"),
    }


async def cancelBooking(booking_id: str, confirmed: bool = False,
                        _context: dict | None = None, **_: Any) -> dict[str, Any]:
    if confirmed is not True:
        return {"cancelled": False, "error": "not_confirmed"}
    bid = norm_booking_id(booking_id or "")
    if not _ID_RE.fullmatch(bid):
        return {"cancelled": False, "error": "invalid_booking_id_format"}
    session_id = (_context or {}).get("session_id")
    email = (_context or {}).get("email") or ""
    doc = await mongo.find_booking_for_caller(bid, email)
    if doc and await mongo.cancel_booking(doc["_id"], session_id=session_id):
        return {"cancelled": True, "booking_id": bid}
    return {"cancelled": False, "error": "could_not_cancel"}


EXECUTORS = {"getBookingDetails": getBookingDetails, "cancelBooking": cancelBooking}