"""
Phase 4/5 — Tool definitions for the AI agent.

Reads cities/hotels/room types/prices live from MongoDB (mongo.py) —
nothing here is hardcoded. The only write in the whole app is
mongo.insert_booking(), called from ai_response.py after
create_booking() below returns successfully.
"""
from __future__ import annotations

import uuid
from math import ceil
from typing import Any

import mongo

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_cities",
            "description": "Cities with active hotels. Call before asking which city; only offer what it returns.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_hotels",
            "description": "Hotels in a given city, once named.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_room_options",
            "description": "Room types and nightly prices at a hotel, once chosen. Only offer what this returns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hotel": {"type": "string"},
                    "city": {"type": "string"},
                },
                "required": ["hotel", "city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_contact_number",
            "description": "Record the confirmed phone number, once, before asking about adults/children.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "confirmed_caller_id": {
                        "type": "boolean",
                        "description": "true if they confirmed the number read back to them, false if they gave a different one",
                    },
                },
                "required": ["number", "confirmed_caller_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_room_count",
            "description": "Check room count is enough (max 2 adults + 1 child per room). Call right after adults/children/rooms are given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "adults": {"type": "integer"},
                    "children": {"type": "integer"},
                    "rooms": {"type": "integer"},
                },
                "required": ["adults", "children", "rooms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_booking",
            "description": "Create the booking. Only after the customer heard it read back and confirmed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "hotel": {"type": "string"},
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "check_in": {"type": "string", "description": "YYYY-MM-DD"},
                    "check_out": {"type": "string", "description": "YYYY-MM-DD"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "adults": {"type": "integer"},
                    "children": {"type": "integer"},
                    "rooms": {"type": "integer"},
                    "room_type": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "city", "hotel", "first_name", "last_name", "check_in",
                    "check_out", "email", "phone", "adults", "children",
                    "rooms", "room_type",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_field_attempt",
            "description": "Record what the customer just said for first_name/last_name/email/check_in/check_out. Call it ONCE per customer message; the system then reads it back and handles their yes/no. Dates as YYYY-MM-DD.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["first_name", "last_name", "email",
                                 "check_in", "check_out"],
                    },
                    "value": {"type": "string"},
                },
                "required": ["field", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_field",
            "description": "Lock in a field the moment the customer confirms its readback. Never ask about it again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["first_name", "last_name", "email",
                                 "hotel", "check_in", "check_out", "room_type"],
                    },
                    "value": {"type": "string"},
                },
                "required": ["field", "value"],
            },
        },
    },
]

def _min_rooms_required(adults: int, children: int) -> int:
    # Business rule: max 2 adults and 1 child per room.
    return max(ceil(adults / 2), children)


async def list_cities() -> dict[str, Any]:
    return {"cities": await mongo.get_cities()}


async def list_hotels(city: str) -> dict[str, Any]:
    city_doc = await mongo.get_city(city)
    if city_doc is None:
        return {"error": "city not found"}
    hotels = await mongo.get_hotels(city_doc["_id"])
    return {"hotels": [h["name"] for h in hotels]}


async def _get_room_options(hotel_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Joins rooms + rateplans for one hotel into [{type, price_per_night,
    max_adults, max_children}, ...]. Uses rates[0] as a flat nightly rate."""
    rooms = await mongo.get_rooms_for_hotel(hotel_doc["_id"])
    rateplan = await mongo.get_rateplan_for_hotel(hotel_doc["_id"])
    rate_by_room_id = {}
    if rateplan:
        for r in rateplan.get("rooms", []):
            rate_by_room_id[r["roomId"]] = r

    options = []
    for room in rooms:
        rate_entry = rate_by_room_id.get(room["_id"])
        if rate_entry is None:
            continue  # no pricing for this room type — don't offer it
        guests_prices = rate_entry.get("prices", {}).get("guests", [])
        if not guests_prices:
            continue
        cheapest = min(guests_prices, key=lambda g: g["rates"][0])
        options.append({
            "type": room["title"]["EN"],
            "price_per_night": cheapest["rates"][0],
            "max_adults": room.get("adults"),
            "max_children": room.get("children"),
        })
    return options


async def get_room_options(hotel: str, city: str) -> dict[str, Any]:
    city_doc = await mongo.get_city(city)
    if city_doc is None:
        return {"error": "city not found"}
    hotel_doc = await mongo.get_hotel(hotel, city_doc["_id"])
    if hotel_doc is None:
        return {"error": "hotel not found"}
    return {"room_types": await _get_room_options(hotel_doc)}

async def confirm_contact_number(number: str, confirmed_caller_id: bool) -> dict[str, Any]:
    return {"number": number, "confirmed_caller_id": confirmed_caller_id}


async def confirm_field(field: str, value: str) -> dict[str, Any]:
    # Just echoes back — the actual effect (writing into
    # session.checklist and session.locked_fields) happens in
    # ai_response._apply_tool_result, same pattern as
    # confirm_contact_number above.
    return {"field": field, "value": value, "confirmed": True}


async def validate_room_count(adults: int, children: int, rooms: int) -> dict[str, Any]:
    minimum = _min_rooms_required(adults, children)
    return {"valid": rooms >= minimum, "minimum_rooms_required": minimum}


async def create_booking(
    city: str, hotel: str, first_name: str, last_name: str,
    check_in: str, check_out: str, email: str, phone: str,
    adults: int, children: int, rooms: int, room_type: list[str] | str,
) -> dict[str, Any]:
    city_doc = await mongo.get_city(city)
    if city_doc is None:
        return {"error": "city not found"}
    hotel_doc = await mongo.get_hotel(hotel, city_doc["_id"])
    if hotel_doc is None:
        return {"error": "hotel not found"}

    options = await _get_room_options(hotel_doc)
    types = [room_type] if isinstance(room_type, str) else list(room_type or [])
    if len(types) != rooms:
        return {"error": "need exactly one room type per room"}
    by_type = {r["type"].lower(): r for r in options}
    chosen = []
    for t in types:
        r = by_type.get(str(t).lower())
        if r is None:
            return {"error": f"room type not available at this hotel: {t}"}
        chosen.append(r)

    minimum = _min_rooms_required(adults, children)
    if rooms < minimum:
        return {"error": f"not enough rooms for this party — at least {minimum} needed"}

    return {
        "booking_id": f"BK-{uuid.uuid4().hex[:8].upper()}",
        "rooms": [{"type": r["type"], "price_per_night": r["price_per_night"]} for r in chosen],
        "total_per_night": sum(r["price_per_night"] for r in chosen),
        "rate_per_night": sum(r["price_per_night"] for r in chosen),
        "currency": "INR",
    }


# REPLACE EXECUTORS with:
EXECUTORS = {
    "list_cities": list_cities,
    "list_hotels": list_hotels,
    "get_room_options": get_room_options,
    "confirm_contact_number": confirm_contact_number,
    "confirm_field": confirm_field,
    "validate_room_count": validate_room_count,
    "create_booking": create_booking,
}