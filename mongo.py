"""
MongoDB access (motor, async) for the booking agent and the support agent.

Reads:  cities, hotels, rooms, rate plans, bookings, support tickets.
Writes: insert_booking      - creates a confirmed booking (called from
                              ai_response.py once a booking is created)
        cancel_booking      - marks a booking cancelled, never deletes
        save_support_ticket - creates a support ticket
        add_ticket_note     - appends a note to a ticket's history
        save_call_log       - one document per support call

Uses motor (Mongo's official async driver) to stay consistent with the
rest of the codebase being asyncio-based. Requires `pip install motor`.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

from config import settings
from session import Session

log = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


# ------------------------------------------------------------- connection
def _get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        # Explicit, short timeouts: a bad/slow connection to a remote
        # Mongo host (India -> wherever MONGODB_URI points) should fail
        # fast and visibly, not silently eat up to PyMongo's default 30s
        # per call while generate_reply() looks like it is "thinking".
        _client = AsyncIOMotorClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
    return _client


def _db():
    return _get_client()[settings.MONGODB_DB_NAME]


def _collection(name: str):
    return _db()[name]


def _bookings_collection():
    return _collection(settings.MONGODB_BOOKINGS_COLLECTION)


def _tickets_collection():
    return _collection("supportTickets")


def _calls_collection():
    return _collection("supportCalls")


async def close() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


# ----------------------------------------------- cities / hotels / rooms
async def get_cities() -> list[str]:
    """Pure read — titles of active cities."""
    cursor = _collection("cities").find({"status": "active"}, {"title": 1})
    return [doc["title"] async for doc in cursor]


async def get_city(title: str) -> dict[str, Any] | None:
    """Pure read — case-insensitive exact match on city title."""
    return await _collection("cities").find_one(
        {"title": {"$regex": f"^{re.escape(title.strip())}$", "$options": "i"}, "status": "active"}
    )


async def get_hotels(city_id: Any) -> list[dict[str, Any]]:
    """Pure read — active hotels in a given city (by cities._id)."""
    cursor = _collection("hotels").find({"cityID": city_id, "status": "active"})
    return [doc async for doc in cursor]


async def get_hotel(name: str, city_id: Any) -> dict[str, Any] | None:
    """Pure read — case-insensitive exact match on hotel name within a city."""
    return await _collection("hotels").find_one(
        {"name": {"$regex": f"^{re.escape(name.strip())}$", "$options": "i"}, "cityID": city_id, "status": "active"}
    )


async def get_all_hotel_names() -> list[str]:
    """Pure read — every active hotel's name, across all cities. Used
    to build the live STT keyterm list (see stt_client.build_keyterms)
    so speech recognition is boosted for the hotels that actually
    exist right now, instead of a hardcoded snapshot going stale."""
    cursor = _collection("hotels").find({"status": "active"}, {"name": 1})
    return [doc["name"] async for doc in cursor]


async def get_active_hotels() -> dict[str, str]:
    """Pure read — {hotel _id as str: name} for every active hotel. Used by
    okf_store to serve knowledge only for hotels that are active here."""
    cursor = _collection("hotels").find({"status": "active"}, {"name": 1})
    return {str(doc["_id"]): doc["name"] async for doc in cursor}


async def get_rooms_for_hotel(hotel_id: Any) -> list[dict[str, Any]]:
    """Pure read — room types at a hotel. rooms.hotelIDs stores the
    hotel's _id as a STRING (not ObjectId) — that's how the source
    data is actually shaped, not a typo here."""
    cursor = _collection("rooms").find({"hotelIDs": str(hotel_id), "status": "active"})
    return [doc async for doc in cursor]


async def get_rateplan_for_hotel(hotel_id: Any) -> dict[str, Any] | None:
    """Pure read — the master rate plan for a hotel."""
    return await _collection("rateplans").find_one(
        {"hotelId": hotel_id, "status": "active", "isMasterPlan": True}
    )


# ------------------------------------------------- booking ID matching
def _norm_booking_id(raw: str) -> str:
    """Spoken IDs arrive as 'bk a1b2 c3d4' or 'BK-A1B2C3D4'. Normalize to the
    stored format: BK- plus 8 uppercase characters."""
    s = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    if s.startswith("BK"):
        s = s[2:]
    return f"BK-{s}"


def canon_booking_id(raw: str) -> str:
    """Comparison key: BK prefix dropped, and letters speech-to-text confuses
    with digits folded together (O/0, I/L/1)."""
    s = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if s.startswith("BK"):
        s = s[2:]
    return s.translate(str.maketrans("OIL", "011"))


def _stored_ids(doc: dict[str, Any]) -> list[str]:
    b = doc.get("booking") or {}
    return [x for x in (doc.get("bookingID"), doc.get("booking_id"), b.get("booking_id")) if x]


_SIMILAR = ["MN", "BDEGPTVCZ", "FSX", "AH8", "KJ"]  # letters speech-to-text mixes up


def _alike(a: str, b: str) -> bool:
    return a == b or any(a in g and b in g for g in _SIMILAR)


def _near(want: str, have: str) -> bool:
    """Same length, at most one differing character, and it is a confusable pair."""
    return (len(want) == len(have)
            and all(_alike(a, b) for a, b in zip(want, have))
            and sum(a != b for a, b in zip(want, have)) <= 1)


# ---------------------------------------------------------------- bookings
async def find_bookings(query: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]:
    """Read multiple bookings matching a filter, e.g.
    {"booking.city": "Ahmedabad"}. Pure read — no write. `limit` caps
    how many documents come back."""
    cursor = _bookings_collection().find(query).limit(limit)
    return [doc async for doc in cursor]


async def get_booking(session_id: str) -> dict[str, Any] | None:
    """Read a single booking by session_id. Pure read — no write."""
    return await _bookings_collection().find_one({"session_id": session_id})


async def find_booking(booking_id: str) -> dict[str, Any] | None:
    """Read one booking by the ID the caller quotes (transcript left out)."""
    return await _bookings_collection().find_one(
        {"booking.booking_id": _norm_booking_id(booking_id)},
        {"_id": 0, "conversation": 0},
    )


async def find_booking_for_caller(booking_id: str, email: str) -> dict[str, Any] | None:
    """Narrow by email first, then match the ID tolerantly, so a fuzzy match
    can only ever hit that caller's own bookings."""
    want = canon_booking_id(booking_id)
    mail = re.sub(r"\s+", "", email or "").lower()
    if not want or "@" not in mail:
        return None
    rx = {"$regex": f"^{re.escape(mail)}$", "$options": "i"}
    docs = await find_bookings({"$or": [{"email": rx}, {"checklist.email": rx}]}, limit=50)
    log.debug(
        "[lookup-debug] candidates_by_email=%d want=%s stored=%s",
        len(docs), want,
        [canon_booking_id(x) for d in docs for x in _stored_ids(d)],
    )
    for doc in docs:
        if want in {canon_booking_id(x) for x in _stored_ids(doc)}:
            return doc
    near = [d for d in docs
            if any(_near(want, canon_booking_id(x)) for x in _stored_ids(d))]
    return near[0] if len(near) == 1 else None


def session_to_booking_doc(session: Session) -> dict[str, Any]:
    """Builds the document stored in the bookings collection."""
    if session.booking is None:
        raise ValueError(f"Session {session.session_id} has no booking data to save.")
    booking_id = session.booking.get("booking_id")
    if not booking_id:
        raise ValueError("booking has no booking_id - refusing to save")
    return {
        "bookingID": booking_id,   # matches the unique bookingID_1 index
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "confirmed_at": session.updated_at.isoformat(),
        "checklist": session.checklist.to_dict(),
        "booking": session.booking,
        "conversation": session.conversation,
    }


async def insert_booking(doc: dict[str, Any]) -> str:
    """Inserts one new confirmed booking document. Call this once, when a
    booking is first confirmed, and never again for that booking."""
    result = await _bookings_collection().insert_one(doc)
    return str(result.inserted_id)


async def cancel_booking(doc_id: Any, session_id: str | None = None) -> bool:
    """Marks a booking cancelled (never deletes). Returns True if it changed."""
    result = await _bookings_collection().update_one(
        {"_id": doc_id, "status": {"$ne": "cancelled"}},
        {"$set": {
            "status": "cancelled",
            "cancelled_at": datetime.now(timezone.utc),
            "cancelled_by": "voice_agent",
            "cancel_session_id": session_id,
        }},
    )
    return result.modified_count == 1


# ----------------------------------------------------------------- tickets
async def save_support_ticket(doc: dict[str, Any]) -> str:
    doc = {**doc, "created_at": datetime.now(timezone.utc), "history": []}
    result = await _tickets_collection().insert_one(doc)
    return str(result.inserted_id)


async def get_support_ticket(ticket_id: str) -> dict[str, Any] | None:
    return await _tickets_collection().find_one({"ticket_id": ticket_id}, {"_id": 0})


async def add_ticket_note(ticket_id: str, note: str) -> bool:
    result = await _tickets_collection().update_one(
        {"ticket_id": ticket_id},
        {"$push": {"history": {
            "at": datetime.now(timezone.utc), "by": "voice_agent", "note": note,
        }}},
    )
    return result.modified_count == 1


# ---------------------------------------------------------------- call log
async def save_call_log(session) -> str:
    """One document per call: outcome, escalation reason, ticket and transcript.
    The transcript contains what the caller said (names, emails), so restrict
    access to this collection."""
    looked_up = session.tool_context.get("getBookingDetails") or {}
    doc = {
        "session_id": session.session_id,
        "started_at": session.created_at,
        "ended_at": datetime.now(timezone.utc),
        "stage": session.stage.value,
        "language": session.language,
        "verified": session.verified_identity,
        "booking_id": looked_up.get("booking_id") if session.verified_identity else None,
        "failed_verifications": session.failed_verifications,
        "backend_failures": session.backend_failures,
        "injection_attempts": session.injection_attempts,
        "guardrail_blocks": session.guardrail_blocks,
        "escalation_reason": session.escalation_reason,
        "handoff_summary": session.handoff_summary,
        "ticket": session.support_ticket.to_dict() if session.support_ticket else None,
        "conversation": session.conversation,
    }
    result = await _calls_collection().insert_one(doc)
    return str(result.inserted_id)