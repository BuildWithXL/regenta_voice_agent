"""
Support tools: tickets, human transfer and hotel FAQ.

ai_response.py only calls EXECUTORS[name](**args). Internal arguments such as
`_context` (session id, verified booking id) are injected by ai_response.py,
never taken from the model.
"""
from __future__ import annotations
import okf
import json
import uuid
from pathlib import Path
from typing import Any

import guardrails
import mongo
from config import settings

VALID_CATEGORIES = ["booking_query", "payment_query", "checkin_logistics",
                    "customer_support", "complaint", "modification"]
VALID_PRIORITIES = ["low", "normal", "high", "urgent"]
HOTEL_TOPICS = ["check_in_time", "check_out_time", "cancellation_policy", "address",
                "contact_phone", "amenities", "parking", "breakfast", "payment_methods",
                "pets_policy", "early_check_in_late_check_out", "id_requirements"]

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "createSupportTicket",
            "description": (
                "Open a ticket when the issue cannot be resolved on the call, or to "
                "record a booking change request (category 'modification', put the "
                "requested change in the description). category must be one of "
                "getSupportCategories -- never invent one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": VALID_CATEGORIES},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": VALID_PRIORITIES},
                },
                "required": ["category", "description", "priority"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getSupportTicket",
            "description": "Fetch the status of a ticket for the verified booking. Internal notes are never read aloud.",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "updateSupportTicket",
            "description": "Add a note from the caller to an existing ticket of the verified booking.",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}, "note": {"type": "string"}},
                "required": ["ticket_id", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getSupportCategories",
            "description": "Fetch the support category list so tickets are classified consistently.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getHotelInfo",
            "description": (
                "Look up general hotel information (times, policies, address, amenities). "
                "Use it for every general hotel question. Pass the hotel name if known."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "enum": HOTEL_TOPICS},
                    "hotel": {"type": "string"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transferToHuman",
            "description": (
                "Hand the call to a human agent with context. Call immediately for a "
                "payment dispute, legal threat or suspected fraud, or when the caller "
                "asks for a person -- never argue them out of it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "urgency": {"type": "string", "enum": ["normal", "high"]},
                    "context_summary": {"type": "string"},
                },
                "required": ["reason", "context_summary"],
            },
        },
    },
]


# ------------------------------------------------------------------ helpers
def _verified_booking_id(_context: dict | None) -> str | None:
    return (_context or {}).get("booking_id")


def _load_hotel_info() -> dict:
    path = Path(settings.HOTEL_INFO_PATH)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------- executors
async def createSupportTicket(category: str, description: str, priority: str,
                              booking_id: str | None = None,
                              _context: dict | None = None, **_: Any) -> dict:
    if category not in VALID_CATEGORIES:
        return {"error": f"invalid category '{category}'"}
    if priority not in VALID_PRIORITIES:
        priority = "normal"
    ctx = _context or {}
    description = guardrails.clean_text(description, 1000)  # stored text: no control chars, capped
    booking_id = ctx.get("booking_id") or booking_id  # verified booking wins
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    await mongo.save_support_ticket({
        "ticket_id": ticket_id,
        "session_id": ctx.get("session_id"),
        "booking_id": booking_id,
        "category": category,
        "description": description,
        "priority": priority,
        "status": "open",
        "language": ctx.get("language"),
        "source": "voice_agent",
    })
    # If the insert raised, the model gets an error and never reads out a ticket ID.
    return {"ticket_id": ticket_id, "status": "open", "booking_id": booking_id}


async def getSupportTicket(ticket_id: str, _context: dict | None = None, **_: Any) -> dict:
    verified = _verified_booking_id(_context)
    doc = await mongo.get_support_ticket(ticket_id)
    if not verified or not doc or doc.get("booking_id") != verified:
        return {"error": "ticket not found for this booking"}
    return {k: doc.get(k) for k in ("ticket_id", "status", "priority", "category", "description")}


async def updateSupportTicket(ticket_id: str, note: str,
                              _context: dict | None = None, **_: Any) -> dict:
    verified = _verified_booking_id(_context)
    doc = await mongo.get_support_ticket(ticket_id)
    if not verified or not doc or doc.get("booking_id") != verified:
        return {"error": "ticket not found for this booking"}
    ok = await mongo.add_ticket_note(ticket_id, guardrails.clean_text(note, 1000))
    return {"ticket_id": ticket_id, "updated": ok}


async def getSupportCategories(**_: Any) -> dict:
    return {"categories": VALID_CATEGORIES}


async def getHotelInfo(topic: str, **_) -> dict:
    doc = okf.get(topic)
    if not doc:
        return {"found": False, "topic": topic}
    return {"found": True, "topic": topic,
            "title": doc["meta"].get("title"), "answer": doc["body"]}

async def transferToHuman(reason: str, context_summary: str, urgency: str = "normal", **_: Any) -> dict:
    # The actual phone redirect happens in main.py once the call is marked TRANSFERRED.
    return {"transferred": True, "reason": reason, "urgency": urgency}


EXECUTORS = {
    "createSupportTicket": createSupportTicket,
    "getSupportTicket": getSupportTicket,
    "updateSupportTicket": updateSupportTicket,
    "getSupportCategories": getSupportCategories,
    "getHotelInfo": getHotelInfo,
    "transferToHuman": transferToHuman,
}
