"""
New-booking flow (booking stage machine + hotel info). ai_response.py
routes calls here; existing bookings go to support_agent.py.

The model is only used to UNDERSTAND what the caller said (pick the
city, pull a name/email/date out of noisy speech). Everything that
decides WHAT HAPPENS NEXT is enforced here in code, because the model
kept skipping steps on its own:

  * name / email / check-in / check-out are never locked until the
    caller says "yes" to a read-back (session.pending, checked in code)
  * guests/rooms are only accepted if the caller actually said numbers
  * the next question after any step is chosen by _current_step(),
    not by the model; an off-step reply ("anything else I can help
    with?") is replaced with the real next question
  * the model can never call create_booking - the whole booking is read
    back and only a "yes" to THAT creates it (and it is saved to
    MongoDB before the caller is told it is confirmed)

generate_reply() is still the conversation loop's single entry point.
"""
from __future__ import annotations

import calendar
import difflib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import voice_guardrails as vg

import groq
from groq import AsyncGroq

import mongo
import okf_store
from ai_tools import EXECUTORS, TOOL_DEFINITIONS
from config import require, settings
from session import Session, Stage

_client: AsyncGroq | None = None

_PROMPT_HEADER = (
    "You are a warm, efficient hotel booking phone agent. Reply must be "
    "at most ONE short sentence plus ONE question - nothing else. Never "
    "repeat a question, never use meta text like [Waiting]. Never say "
    "'anything else I can help with' - the booking is not finished until "
    "the system says so. "
    "Today's date is {today} ({weekday}). "
    "A bare greeting alone (hi, how are you) is NOT booking intent - "
    "reply warmly in one line and ask if they would like to book. Start "
    "the step below only once they show real intent.\n"
    "Never invent a city/hotel/room/price - only offer what the tools "
    "actually returned. If their answer sounds like a close phonetic "
    "match to a real option (STT mishearing), treat it as that option.\n"
    "Messages may be prefixed [transcription uncertain] - if you can "
    "make sense of it respond normally, else say you did not catch it.\n"
    "Email: NEVER ask them to spell it - normal spoken words only. "
    "at / at the rate means @, dot means . . Assume .com for known "
    "providers (gmail/yahoo/outlook/hotmail/icloud/rediffmail).\n"
    "NEVER call a tool with a value the customer did not just say. "
    "NEVER call confirm_field for names, email or dates - the system "
    "reads those back and handles the yes/no itself.\n"
    "LANGUAGE: caller may mix English/Hindi/Gujarati - understand all, "
    "ALWAYS reply in simple English. Tool arguments (city, hotel, room "
    "type, dates) stay in their exact English/canonical form.\n"
)

# Only the current step's instructions are sent, keyed by the first
# missing checklist field (see _system_prompt).
_STEP_BLOCKS: dict[str, str] = {
    "city": (
        "STEP: if you haven't already, call list_cities and ask which "
        "city (only offer what it returns). The MOMENT they name one, "
        "call list_hotels for that city right away in that same reply. "
        "Do not call list_cities again once they've answered.\n"
    ),
    "hotel": (
        "STEP: the hotels for the city were already listed. When they "
        "name one, call confirm_field(hotel, exact name from the list) "
        "in that same reply.\n"
    ),
    "first_name": (
        "STEP: ask their first name. The moment they say one, call "
        "record_field_attempt(first_name, value) ONCE and stop - the "
        "system reads it back and handles yes/no.\n"
    ),
    "last_name": (
        "STEP: ask their last name. The moment they say one, call "
        "record_field_attempt(last_name, value) ONCE and stop.\n"
    ),
    "check_in": (
        "STEP: ask the check-in date. The moment they say one, call "
        "record_field_attempt(check_in, YYYY-MM-DD) ONCE and stop.\n"
    ),
    "check_out": (
        "STEP: ask the check-out date. The moment they say one, call "
        "record_field_attempt(check_out, YYYY-MM-DD) ONCE and stop.\n"
    ),
    "email": (
        "STEP: ask their email address. The moment they say it, call "
        "record_field_attempt(email, value) ONCE and stop.\n"
    ),
    "phone": (
        "STEP: the caller's number is {caller_number}. If it is a real "
        "number, ask if it is the best number to reach them on; if they "
        "give a different number use that. Call confirm_contact_number "
        "only once they have actually said a phone number or yes.\n"
    ),
    "phone_confirmed": (
        "STEP: the caller's number is {caller_number}. If it is a real "
        "number, ask if it is the best number to reach them on; if they "
        "give a different number use that. Call confirm_contact_number "
        "only once they have actually said a phone number or yes.\n"
    ),
    "adults": "STEP: ask how many adults (guests aged 12 and above).\n",
    "children": "STEP: ask how many children under 12.\n",
    "rooms": "STEP: ask how many rooms they would like.\n",
    "room_type": (
        "STEP: the room types were already listed. The customer picks one "
        "room type PER ROOM. When they name one, call "
        "confirm_field(room_type, exact name from the list) once.\n"
    ),
    "_done": "STEP: every detail is collected.\n",
}

# ----------------------------------------------------------------------
# Field order / questions / keywords - the ONE source of "where are we"
# ----------------------------------------------------------------------
_FIELD_ORDER = [
    "city", "hotel", "first_name", "last_name", "check_in", "check_out",
    "email", "phone", "phone_confirmed", "adults", "children", "rooms",
    "room_type",
]

_STEP_QUESTIONS = {
    "city": "Which city would you like to stay in?",
    "hotel": "Which hotel would you like to stay at?",
    "first_name": "What's your first name?",
    "last_name": "What's your last name?",
    "check_in": "What's your check-in date?",
    "check_out": "What's your check-out date?",
    "email": "What's your email address?",
    "phone": "What's the best phone number to reach you on?",
    "phone_confirmed": "What's the best phone number to reach you on?",
    "party": "How many adults, how many children, and how many rooms would you like?",
    "adults": "How many adults will be staying? Guests aged 12 and above count as adults.",
    "children": "And how many children under 12 are travelling?",
    "rooms": "How many rooms would you like?",
    "room_type": "Which room type would you like?",
}

# A reply at a step must contain one of these (or name a real option).
_STEP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "city": ("city",),
    "hotel": ("hotel",),
    "first_name": ("first name",),
    "last_name": ("last name", "surname"),
    "check_in": ("check-in", "check in", "checkin"),
    "check_out": ("check-out", "check out", "checkout"),
    "email": ("email", "e-mail"),
    "phone": ("phone", "number", "reach you"),
    "phone_confirmed": ("phone", "number", "reach you"),
    "party": ("adult", "children", "child", "rooms", "guests"),
    "adults": ("adult", "guests"),
    "children": ("child", "children", "kids"),
    "rooms": ("room",),
    "room_type": ("room",),
}

# Strong keywords of LATER steps - a reply at an earlier personal-detail
# step that mentions these is trying to skip ahead.
_LATER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "last_name": ("last name", "surname"),
    "check_in": ("check-in", "check in"),
    "check_out": ("check-out", "check out"),
    "email": ("email",),
    "phone": ("phone number", "contact number"),
    "adults": ("adults",),
    "children": ("children",),
    "rooms": ("how many rooms",),
    "room_type": ("room type",),
}

_PERSONAL_STEPS = ["first_name", "last_name", "check_in", "check_out",
                   "email", "phone", "phone_confirmed", "adults", "children", "rooms"]

# Fields locked only by a "yes" to a read-back (done in code).
_CONFIRM_FIELDS = {"first_name", "last_name", "email", "check_in", "check_out"}
_RETRY_FIELDS = {"first_name", "last_name", "email"}
_MAX_FIELD_ATTEMPTS = 3   # real-looking guesses before a name/email is force-accepted


def _is_field_missing(checklist, field_name: str) -> bool:
    if field_name == "phone_confirmed":
        return getattr(checklist, "phone_confirmed", None) is not True
    if field_name == "room_type":
        types = checklist.room_type
        return not isinstance(types, list) or len(types) < (checklist.rooms or 1)
    return getattr(checklist, field_name, None) is None


def _current_step(session: Session) -> str:
    for f in _FIELD_ORDER:
        if _is_field_missing(session.checklist, f):
            return f
    return "_done"


# ----------------------------------------------------------------------
# Small text helpers
# ----------------------------------------------------------------------
def _spoken_list(items) -> str:
    """'A, B and C' - reads naturally over the phone (uses ';' when the
    items themselves contain commas, e.g. 'Vista Haven, Goa')."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    sep = "; " if any("," in i for i in items) else ", "
    return sep.join(items[:-1]) + " and " + items[-1]


def _mentions_any(text: str, names) -> bool:
    low = text.lower()
    return any(n and str(n).lower() in low for n in names)


def _strip_markers(text: str) -> str:
    """Drops the '[caller spoke Hindi]' / '[transcription uncertain]'
    prefixes main.py adds, so yes/no and values are read from the words."""
    return re.sub(r"^(\s*\[[^\]]*\]\s*)+", "", text).strip()


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _digits_spoken(number: str) -> str:
    return " ".join(re.sub(r"\D", "", str(number)))


_YES = {"yes", "yeah", "yep", "yup", "yea", "correct", "right", "sure", "ok",
        "okay", "haan", "han", "ha", "ji", "theek", "thik", "absolutely",
        "exactly", "confirm", "confirmed", "perfect"}
_NO = {"no", "nope", "nah", "nahi", "nahin", "wrong", "incorrect", "not"}


def _yes_no(text: str) -> str | None:
    """'yes' / 'no' for a short plain answer; 'correction' when they said
    no AND gave something new (e.g. 'no my name is Drashti'); else None."""
    toks = re.findall(r"[a-z']+", text.lower())
    if not toks or len(toks) > 6:
        return None
    has_yes = any(t in _YES for t in toks)
    has_no = any(t in _NO for t in toks)
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no" if len(toks) <= 3 else "correction"
    return None


_NUM_WORDS = {"zero": 0, "no": 0, "none": 0, "nil": 0, "one": 1, "two": 2,
              "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10}


def _numbers_in(text: str) -> set[int]:
    nums: set[int] = set()
    for tok in re.findall(r"[a-z]+|\d+", text.lower()):
        if tok.isdigit():
            nums.add(int(tok))
        elif tok in _NUM_WORDS:
            nums.add(_NUM_WORDS[tok])
    return nums

# ---- guest / room counts ----------------------------------------------
def _min_rooms(adults: int, children: int) -> int:
    # Business rule: max 2 adults and 1 child per room.
    return max(math.ceil(adults / 2), children, 1)


_HOMOPHONES = {"to": 2, "too": 2, "tu": 2, "for": 4, "won": 1, "tree": 3, "ate": 8}
_NUM_ALT = "|".join(sorted([*_NUM_WORDS, r"\d+"], key=len, reverse=True))
_ADULT_NOUNS = "adults?|people|persons?|guests?"
_CHILD_NOUNS = "children|child|kids?"
_ROOM_NOUNS = "rooms?"


def _count_for(text: str, nouns: str) -> int | None:
    """'2 adults' / 'no children' / 'three rooms' -> the number before the noun."""
    m = re.search(rf"\b({_NUM_ALT})\s*(?:{nouns})\b", text.lower())
    if not m:
        return None
    tok = m.group(1)
    return int(tok) if tok.isdigit() else _NUM_WORDS[tok]


def _single_number(text: str) -> int | None:
    """The one number in a short answer like 'two' / '3' / 'no'."""
    toks = re.findall(r"[a-z]+|\d+", text.lower())
    if len(toks) == 1 and toks[0] in _HOMOPHONES:
        return _HOMOPHONES[toks[0]]
    nums = _numbers_in(text)
    return next(iter(nums)) if len(nums) == 1 else None


def _add_room_types(session: Session, types: list[str], fill_all: bool = False) -> None:
    """Adds room types (one per room, never more than 'rooms')."""
    c = session.checklist
    total = c.rooms or 1
    chosen = list(c.room_type or [])
    for t in types[: max(total - len(chosen), 0)]:
        chosen.append(t)
    if fill_all and chosen:
        chosen += [chosen[-1]] * (total - len(chosen))
    c.room_type = chosen
    session.touch()


def _capacity_problem(session: Session) -> str | None:
    """None if the chosen rooms can hold the party, else what to tell them."""
    c = session.checklist
    caps = [session.room_capacity.get(t, (2, 1)) for t in (c.room_type or [])]
    max_a, max_c = sum(x[0] for x in caps), sum(x[1] for x in caps)
    if c.adults > max_a or c.children > max_c:
        return (f"Those rooms hold at most {_plural(max_a, 'adult', 'adults')} and "
                f"{_plural(max_c, 'child', 'children')} in total, so they can't fit your group.")
    return None

# ---- emails -----------------------------------------------------------
_KNOWN_EMAIL_PROVIDERS = ("gmail", "yahoo", "outlook", "hotmail", "icloud", "rediffmail")
_EMAIL_RE = re.compile(r"^[a-z0-9._%+\-]+@[a-z0-9\-]+(\.[a-z0-9\-]+)+$")


def _normalize_email(raw: str) -> str:
    """'Patel Krisha M at gmail dot com' -> 'patelkrisham@gmail.com'."""
    s = raw.strip().lower()
    if "@" not in s:
        s = re.sub(r"\s+at the rate( of)?\s+|\s+at\s+(?!.*\s+at\s+)", "@", s)
    s = re.sub(r"\s+dot\s+", ".", s)
    s = re.sub(r"\s+(underscore|under score)\s+", "_", s)
    s = re.sub(r"\s+(dash|hyphen)\s+", "-", s)
    s = re.sub(r"\s+", "", s)
    if "@" in s:
        local, _, domain = s.partition("@")
        if "." not in domain and domain in _KNOWN_EMAIL_PROVIDERS:
            domain += ".com"
        s = f"{local}@{domain}"
    return s


def _is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email)) and vg.verify_email(email)


def _spoken_email(email: str) -> str:
    return email.replace("@", " at ").replace(".", " dot ")


# ---- names ------------------------------------------------------------
_NAME_STOPWORDS = {
    "what", "is", "yes", "no", "yeah", "nope", "hello", "hi", "sorry",
    "pardon", "okay", "ok", "my", "name", "last", "first", "the", "a", "it",
    "why", "how", "who", "when", "where", "which", "do", "you", "need",
    "can", "please", "repeat", "again",
}


def _normalize_name(raw: str) -> str:
    """'p a t e l' / 'P. A. T. E. L.' -> 'Patel'; 'patel' -> 'Patel'."""
    s = re.sub(r"[.,]", "", str(raw))
    s = re.sub(r"\s+", " ", s.strip())
    parts = s.split(" ")
    if len(parts) > 1 and all(len(p) == 1 for p in parts):
        s = "".join(parts)
    return s.title()


def _is_valid_name(name: str) -> bool:
    """Rejects digits, symbols, 1-letter junk and filler like 'What Is'."""
    if not (2 <= len(name) <= 40):
        return False
    if not all(ch.isalpha() or ch in " '-" for ch in name):
        return False
    return vg.verify_name_part(name) and not all(w in _NAME_STOPWORDS for w in name.lower().split())


def _extract_name(clean: str) -> str:
    """Fallback only: pulls a name out of 'my first name is Drishti'."""
    s = re.sub(
        r"(?i)\b(my\s+)?(first\s+|last\s+)?(sur)?name\s+is\b|\b(it'?s|it\s+is|this\s+is|i\s+am|i'm)\b",
        " ", clean)
    return _normalize_name(s)


# ---- dates ------------------------------------------------------------
_MONTHS: dict[str, int] = {}
for _i, _m in enumerate(calendar.month_name):
    if _m:
        _MONTHS[_m.lower()] = _i
        _MONTHS[_m.lower()[:3]] = _i
_MONTHS["sept"] = 9


def _today():
    return datetime.now().date()


def _parse_spoken_date(text: str):
    """'10th October' / 'October 10' / '10 Oct 2026' / '2026-10-10' -> date."""
    low = text.lower().strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", low)
    if m:
        try:
            return datetime(int(m[1]), int(m[2]), int(m[3])).date()
        except ValueError:
            return None
    day = month = year = None
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s+)?([a-z]{3,9})\b(?:\s*,?\s*(\d{4}))?", low)
    if m and m[2] in _MONTHS:
        day, month, year = int(m[1]), _MONTHS[m[2]], m[3]
    else:
        m = re.search(r"\b([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:\s*,?\s*(\d{4}))?", low)
        if m and m[1] in _MONTHS:
            day, month, year = int(m[2]), _MONTHS[m[1]], m[3]
    if day is None:
        return None
    today = _today()
    y = int(year) if year else today.year
    try:
        d = datetime(y, month, day).date()
        if not year and d < today:
            d = datetime(y + 1, month, day).date()
    except ValueError:
        return None
    return d


def _fmt_date(iso: str) -> str:
    d = datetime.strptime(iso, "%Y-%m-%d")
    return f"{d.day} {d:%B %Y}"


def _validate_field(session: Session, field_name: str, raw: str, strict: bool = False):
    """-> (normalized_value, ok, message_if_not_ok). strict=True is the
    stricter filter used when WE extract the value from plain speech."""
    if field_name in ("first_name", "last_name"):
        v = _extract_name(raw) if strict else _normalize_name(raw)
        words = v.split()
        if strict and (not (1 <= len(words) <= 2)
                       or any(w.lower() in _NAME_STOPWORDS for w in words)):
            return v, False, ""
        return v, _is_valid_name(v), ""
    if field_name == "email":
        v = _normalize_email(raw)
        return v, _is_valid_email(v), ""
    if field_name in ("check_in", "check_out"):
        d = _parse_spoken_date(raw)
        if d is None:
            return raw, False, ""
        iso = d.strftime("%Y-%m-%d")
        if d < _today():
            return iso, False, "That date has already passed."
        if field_name == "check_out" and session.checklist.check_in:
            ci = datetime.strptime(session.checklist.check_in, "%Y-%m-%d").date()
            if d <= ci:
                return iso, False, "Check-out has to be after your check-in date."
        return iso, True, ""
    return raw, True, ""


def _match_option(value: str, options) -> str | None:
    """Exact / substring / close-spelling match against real options."""
    low = {str(o).lower(): o for o in options or []}
    v = value.lower().strip()
    if v in low:
        return low[v]
    for k, o in low.items():
        if v and (v in k or k in v):
            return o
    close = difflib.get_close_matches(v, list(low), n=1, cutoff=0.6)
    return low[close[0]] if close else None


# ----------------------------------------------------------------------
# Prompt building
# ----------------------------------------------------------------------
def _confirmed_summary(session: Session) -> str:
    c = session.checklist
    parts = []
    if c.city:
        parts.append(f"city={c.city}")
    if c.hotel:
        parts.append(f"hotel={c.hotel}")
    if "first_name" in session.locked_fields:
        parts.append(f"first_name={c.first_name}")
    if "last_name" in session.locked_fields:
        parts.append(f"last_name={c.last_name}")
    if c.check_in:
        parts.append(f"check_in={c.check_in}")
    if c.check_out:
        parts.append(f"check_out={c.check_out}")
    if "email" in session.locked_fields:
        parts.append(f"email={c.email}")
    if c.phone_confirmed:
        parts.append(f"phone={c.phone}")
    if c.adults is not None:
        parts.append(f"adults={c.adults}")
    if c.children is not None:
        parts.append(f"children={c.children}")
    if c.rooms is not None:
        parts.append(f"rooms={c.rooms}")
    if c.room_type:
        parts.append(f"room_type={c.room_type}")
    if not parts:
        return ""
    step = _current_step(session)
    return ("\nCONFIRMED SO FAR (do not re-ask any of these): " + ", ".join(parts)
            + f". NEXT STEP: ask for {step}.\n")


def _system_prompt(session: Session) -> str:
    now = datetime.now(timezone.utc)
    caller_number = session.caller_number or "(not available - ask them for their number)"
    header = _PROMPT_HEADER.format(today=now.strftime("%Y-%m-%d"), weekday=now.strftime("%A"))
    step = _STEP_BLOCKS[_current_step(session)].format(caller_number=caller_number)
    return header + "NEXT QUESTION - ask ONLY about this, nothing else:\n" + step + _confirmed_summary(session)


_FALLBACK_REPLY = (
    "Sorry, I'm having trouble on my end right now - could you give me "
    "just a moment, or would you like me to transfer you to a team member?"
)
_MAX_TOOL_ROUNDS = 4
_MAX_HISTORY_MESSAGES = 8


def _strip_repeated_sentences(text: str) -> str:
    parts = re.split(r'(?<=[.?!])\s*', text.strip())
    seen, kept = set(), []
    for part in parts:
        key = part.lower().strip()
        if key and key in seen:
            continue
        seen.add(key)
        kept.append(part)
    return " ".join(kept).strip()


def _truncate_after_first_question(text: str) -> str:
    idx = text.find("?")
    return text.strip() if idx == -1 else text[: idx + 1].strip()


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=require("GROQ_API_KEY"))
    return _client


# ----------------------------------------------------------------------
# Applying tool results to the session
# ----------------------------------------------------------------------
async def _apply_tool_result(
    session: Session, name: str, args: dict[str, Any], output: dict[str, Any]
) -> None:
    if name == "list_cities":
        cities = output.get("cities") if isinstance(output, dict) else None
        if cities:
            session.available_cities = cities
    elif name == "list_hotels":
        if isinstance(output, dict) and "error" in output:
            return  # unknown city - don't lock it in or move to the hotel step
        session.update_checklist(city=args.get("city"))
        names = output.get("hotels") if isinstance(output, dict) else None
        session.available_hotels = [
            (h.get("name") if isinstance(h, dict) else h) for h in (names or [])
        ]
    elif name == "get_room_options":
        types = output.get("room_types") if isinstance(output, dict) else None
        if types:
            session.available_room_types = [
                (t.get("type") if isinstance(t, dict) else t) for t in types
            ]
            session.room_prices = {
                t.get("type"): t.get("price_per_night")
                for t in types if isinstance(t, dict)
            }
            session.room_capacity = {
                t.get("type"): (t.get("max_adults") or 2, t.get("max_children") or 1)
                for t in types if isinstance(t, dict)
            }
    elif name == "confirm_contact_number":
        session.update_checklist(phone=args.get("number"), phone_confirmed=True)
    elif name == "confirm_field":
        field_name = args.get("field")
        if field_name in {"first_name", "last_name", "email",
                          "hotel", "check_in", "check_out", "room_type"}:
            if field_name == "hotel":
                session.available_room_types = None   # rooms belong to a hotel
                session.room_prices = {}
                session.room_capacity = {}
                session.checklist.room_type = None
            if field_name == "room_type":
                _add_room_types(session, [args.get("value")])   # one more room
                return
            session.update_checklist(**{field_name: args.get("value")})
            session.locked_fields.add(field_name)
    elif name == "validate_room_count":
        if output.get("valid"):
            session.update_checklist(
                adults=args.get("adults"),
                children=args.get("children"),
                rooms=args.get("rooms"),
            )
    elif name == "create_booking":
        session.update_checklist(
            **{k: v for k, v in args.items() if hasattr(session.checklist, k)}
        )
        if "error" in output:
            return
        session.booking = output
        booking_id = output.get("booking_id")
        try:
            await mongo.insert_booking(mongo.session_to_booking_doc(session))
        except Exception as exc:
            print(f"[ai_response] booking {booking_id} could not be saved to MongoDB: {exc}")
            session.booking = None
            session.set_stage(Stage.AWAITING_CONFIRMATION)
            output.clear()
            output["error"] = "booking_save_failed"
            return
        session.set_stage(Stage.COMPLETED)


# ----------------------------------------------------------------------
# Deterministic questions / read-backs
# ----------------------------------------------------------------------
def _room_listing(session: Session) -> str:
    prices = getattr(session, "room_prices", None) or {}
    parts = []
    for t in session.available_room_types or []:
        p = prices.get(t)
        parts.append(f"{t} at {p:,.0f} rupees a night" if isinstance(p, (int, float)) else str(t))
    return _spoken_list(parts)


def _step_question(session: Session, step: str) -> str:
    """The question for a step, naming the real options when we have them."""
    if step == "city" and session.available_cities:
        return f"We have hotels in {_spoken_list(session.available_cities)}. Which city would you like?"
    if step == "hotel" and session.available_hotels:
        return (f"In {session.checklist.city} we have {_spoken_list(session.available_hotels)}. "
                f"Which hotel would you like?")
    c = session.checklist
    if step == "rooms":
        need = _min_rooms(c.adults, c.children)
        if need == 1:
            return "Your group fits in 1 room, but you can book more. How many rooms would you like?"
        return (f"For {_plural(c.adults, 'adult', 'adults')} and "
                f"{_plural(c.children, 'child', 'children')} you need at least "
                f"{need} rooms. How many rooms would you like?")
    if step == "room_type" and session.available_room_types:
        total = c.rooms or 1
        lead = "We have" if total == 1 else f"For room {len(c.room_type or []) + 1} of {total}, we have"
        return f"{lead} {_room_listing(session)}. Which room type would you like?"
    if step in ("phone", "phone_confirmed"):
        return _phone_question(session)
    return _STEP_QUESTIONS[step]


async def _prefetch(session: Session, step: str) -> None:
    """Loads the list a step needs from MongoDB if we don't have it yet."""
    c = session.checklist
    try:
        if step == "city" and not session.available_cities:
            out = await EXECUTORS["list_cities"]()
            await _apply_tool_result(session, "list_cities", {}, out)
        elif step == "hotel" and not session.available_hotels and c.city:
            out = await EXECUTORS["list_hotels"](city=c.city)
            await _apply_tool_result(session, "list_hotels", {"city": c.city}, out)
        elif step == "room_type" and not session.available_room_types and c.hotel and c.city:
            out = await EXECUTORS["get_room_options"](hotel=c.hotel, city=c.city)
            await _apply_tool_result(session, "get_room_options", {"hotel": c.hotel, "city": c.city}, out)
    except Exception as exc:
        print(f"  [prefetch] could not load data for step {step}: {exc}")


async def _next_question(session: Session) -> str:
    if _menu_pending(session):                 # hotel just chosen: offer the options first
        return _menu_prompt(session)
    step = _current_step(session)
    if step == "_done":
        problem = _capacity_problem(session)
        if problem:
            session.checklist.room_type = None
            return problem + " " + await _next_question(session)
        return _final_readback(session)
    await _prefetch(session, step)
    return _step_question(session, step)


def _readback(field_name: str, value: str) -> str:
    if field_name == "first_name":
        return f"Is your first name {value}?"
    if field_name == "last_name":
        return f"Is your last name {value}?"
    if field_name == "email":
        return f"Is your email {_spoken_email(value)}?"
    if field_name == "check_in":
        return f"Is your check-in date {_fmt_date(value)}?"
    return f"Is your check-out date {_fmt_date(value)}?"


def _reask_after_no(session: Session, step: str) -> str:
    if step in ("first_name", "last_name"):
        label = "first" if step == "first_name" else "last"
        if session.field_attempts.get(step, 0) >= 2:
            return f"Sorry about that. Could you spell your {label} name letter by letter?"
        return f"Sorry about that. Could you say your {label} name again, clearly?"
    if step == "email":
        return "Sorry about that. Could you say your email address again?"
    return f"Sorry about that. {_STEP_QUESTIONS[step]}"


def _junk_reply(session: Session, step: str, msg: str = "") -> str:
    bad = session.field_attempts.get(f"{step}:invalid", 0)
    if step in ("first_name", "last_name") and bad >= 2:
        label = "first" if step == "first_name" else "last"
        return f"Sorry, I still didn't catch that. Could you spell your {label} name letter by letter?"
    prefix = msg if msg else "Sorry, I didn't catch that."
    return f"{prefix} {_STEP_QUESTIONS[step]}"

# ----------------------------------------------------------------------
# Phone number (first thing on the call) - all in code, no model
# ----------------------------------------------------------------------
_DIGIT_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_REPEAT_WORDS = {"double": 2, "triple": 3}


def _normalize_phone(raw) -> str | None:
    """Any spelling of a number -> 10 digits, or None if it isn't one."""
    digits = re.sub(r"\D", "", str(raw or ""))
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits if len(digits) == 10 else None


def _extract_phone(text: str) -> str | None:
    """'98765 43210' / '9 8 7 6 ...' / 'nine eight double seven ...' -> digits."""
    out, repeat = [], 1
    for tok in re.findall(r"[a-z]+|\d+", text.lower()):
        if tok in _REPEAT_WORDS:
            repeat = _REPEAT_WORDS[tok]
        elif tok.isdigit():
            out.append(tok * repeat if len(tok) == 1 else tok)
            repeat = 1
        elif tok in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[tok] * repeat)
            repeat = 1
    return _normalize_phone("".join(out))


def _phone_state(session: Session) -> str:
    """Current phone sub-state; the first time the phone step is reached
    (or after it was reset) it starts at asked_best / need_number."""
    if session.phone_state in (None, "done"):
        session.phone_state = ("asked_best" if vg.verify_phone(session.caller_number)
                               else "need_number")
    return session.phone_state


def _phone_question(session: Session) -> str:
    if _phone_state(session) == "asked_best":
        return "Is this your best number to reach you on?"
    if session.phone_state == "confirm_number" and session.phone_candidate:
        return f"Your number is {_digits_spoken(session.phone_candidate)}. Is that correct?"
    return "What's your preferred mobile number?"


OPENING_LINE = "Thank you for calling. How can I help you today?"


def opening_line(session: Session) -> str:
    """First thing the caller hears - always the same."""
    session.add_message("assistant", OPENING_LINE)
    return OPENING_LINE


async def _handle_phone(session: Session, clean: str) -> str:
    state = _phone_state(session)
    ans = _yes_no(clean)
    said = _extract_phone(clean)

    if said and said == session.phone_candidate and state == "confirm_number":
        ans, said = "yes", None            # they just repeated the number we read back

    if said and not vg.verify_phone(said):  # fake / invalid number (guardrail)
        session.phone_state, session.phone_candidate = "need_number", None
        return "That doesn't look like a valid mobile number. " + _phone_question(session)

    if said:                               # they gave a number -> read it back
        session.phone_candidate = said
        session.phone_state = "confirm_number"
        return _phone_question(session)

    if state == "asked_best":
        if ans == "yes":                   # "yes" -> read the caller's number back
            session.phone_candidate = _normalize_phone(session.caller_number)
            session.phone_state = "confirm_number"
            return _phone_question(session)
        if ans in ("no", "correction"):    # "no" -> ask for the preferred number
            session.phone_state = "need_number"
            return "No problem. " + _phone_question(session)
        return "Sorry, I didn't catch that. " + _phone_question(session)

    if state == "confirm_number":
        if ans == "yes":                   # confirmed -> continue the normal flow
            session.update_checklist(phone=session.phone_candidate, phone_confirmed=True)
            session.locked_fields.add("phone")
            session.phone_state, session.phone_candidate = "done", None
            session.field_attempts.pop("phone:invalid", None)
            return "Thank you. " + await _next_question(session)
        if ans in ("no", "correction"):
            session.phone_state, session.phone_candidate = "need_number", None
            return "Sorry about that. " + _phone_question(session)
        return "Sorry, I didn't catch that. " + _phone_question(session)

    # need_number, but no usable number in what they said
    session.phone_state = "need_number"
    bad = session.field_attempts.get("phone:invalid", 0) + 1
    session.field_attempts["phone:invalid"] = bad
    if bad >= 2:
        return "Sorry, I still didn't catch it. Please say the ten digits slowly, one by one."
    return "Sorry, I didn't catch that number. " + _phone_question(session)

# ----------------------------------------------------------------------
# Final read-back / booking / changes (all in code - never the model)
# ----------------------------------------------------------------------
def _final_readback(session: Session) -> str:
    c = session.checklist
    session.readback_done = True
    session.awaiting_change = False
    session.set_stage(Stage.AWAITING_CONFIRMATION)
    types = list(c.room_type or [])
    if c.rooms == 1:
        rooms_txt = types[0]
    elif len(set(types)) == 1:
        rooms_txt = f"{c.rooms} rooms, all {types[0]}"
    else:
        rooms_txt = f"{c.rooms} rooms: {_spoken_list(types)}"
    total = sum((session.room_prices or {}).get(t) or 0 for t in types)
    cost = f" That comes to {total:,.0f} rupees a night in total." if total else ""
    return (
        f"Let me confirm: {c.first_name} {c.last_name}, {c.hotel}, from "
        f"{_fmt_date(c.check_in)} to {_fmt_date(c.check_out)}, {rooms_txt}, "
        f"for {_plural(c.adults, 'adult', 'adults')} and "
        f"{_plural(c.children, 'child', 'children')}.{cost} Is everything correct?"
    )

_CHANGE_KEYS: list[tuple[tuple[str, ...], list[str]]] = [
    # longest phrases first inside each group; most specific groups first
    (("first name",), ["first_name"]),
    (("last name", "surname"), ["last_name"]),
    (("hotel name", "hotel"), ["hotel", "room_type"]),
    (("city",), ["city", "hotel", "room_type"]),
    (("name",), ["first_name", "last_name"]),
    (("email", "e-mail"), ["email"]),
    (("phone number", "phone", "mobile", "contact", "my number"), ["phone", "phone_confirmed"]),
    (("check-in date", "check in date", "arrival date", "check-in", "check in", "checkin", "arrival"), ["check_in"]),
    (("check-out date", "check out date", "departure date", "check-out", "check out", "checkout", "departure"), ["check_out"]),
    (("dates", "date"), ["check_in", "check_out"]),
    (("room type", "type of room", "type"), ["room_type"]),
    (("number of adults", "number of children", "number of guests", "adults", "adult", "children", "child", "guests", "guest"),
     ["adults", "children", "rooms"]),
    (("number of rooms", "rooms", "room"), ["rooms"]),   # keeps room types already chosen
]


def _fields_to_change(text: str) -> list[str]:
    low = text.lower()
    found: list[str] = []
    for phrases, fields in _CHANGE_KEYS:
        hit = next((p for p in phrases if p in low), None)
        if hit:
            low = low.replace(hit, " ")
            found += [f for f in fields if f not in found]
    return found


def _reset_fields(session: Session, fields: list[str]) -> None:
    for f in fields:
        setattr(session.checklist, f, None)
        session.locked_fields.discard(f)
        session.field_attempts.pop(f, None)
        session.field_attempts.pop(f"{f}:invalid", None)
    if "city" in fields:
        session.available_hotels = None
    if "hotel" in fields or "city" in fields:
        session.available_room_types = None
        session.room_prices = {}
    if "phone" in fields or "phone_confirmed" in fields:
        session.phone_state = "need_number"      # changing it = ask for a new number
        session.phone_candidate = None
    session.pending = None
    session.readback_done = False
    session.awaiting_change = False
    session.set_stage(Stage.COLLECTING_INFO)

async def _create_booking_now(session: Session) -> str:
    c = session.checklist
    args = dict(
        city=c.city, hotel=c.hotel, first_name=c.first_name, last_name=c.last_name,
        check_in=c.check_in, check_out=c.check_out, email=c.email, phone=c.phone,
        adults=c.adults, children=c.children, rooms=c.rooms, room_type=c.room_type,
    )
    bad = vg.check_booking_ready(c, getattr(session, "available_hotels", None),
                                 getattr(session, "available_room_types", None))
    if bad:                                        # guardrail: never save an invalid booking
        print(f"  [guardrail] booking blocked before save, re-asking: {bad}")
        _reset_fields(session, bad + (["phone_confirmed"] if "phone" in bad else []))
        return "Sorry, I need to re-check some of your details. " + await _next_question(session)
    session.set_stage(Stage.BOOKING)
    try:
        output = await EXECUTORS["create_booking"](**args)
    except Exception as exc:
        print(f"[ai_response] create_booking raised: {exc}")
        session.set_stage(Stage.AWAITING_CONFIRMATION)
        return "I'm sorry, I couldn't complete the booking just now. Shall I try again?"
    if "error" in output:
        print(f"[ai_response] create_booking failed: {output}")
        session.set_stage(Stage.AWAITING_CONFIRMATION)
        return "I'm sorry, I couldn't complete the booking just now. Shall I try again?"
    await _apply_tool_result(session, "create_booking", args, output)
    if session.stage != Stage.COMPLETED:        # MongoDB save failed
        return "I'm sorry, I couldn't save your booking just now. Shall I try again?"
    session.set_stage(Stage.POST_BOOKING)          # the call continues; not an end stage
    session.hotel_menu.update(mode=None, queue=[])
    return (f"Your booking is confirmed. Your booking ID is {output['booking_id']}. "
            f"The total is {output['total_per_night']:,.0f} rupees a night, payable at the "
            f"hotel. " + _POST_OPTIONS)


_CHANGE_WORDS = re.compile(
    r"\b(change|add|remove|edit|update|modify|more|extra|another|instead|reduce|wrong|different|want|need)\b")
_ADD_WORDS = re.compile(r"\b(add|another|extra|more)\b")
_WHICH_DETAIL = "Which detail would you like to change - name, dates, email, phone, guests, rooms or room type?"


async def _apply_change(session: Session, clean: str, fields: list[str]) -> str:
    """Resets the detail(s) the caller wants to change and asks for them again."""
    c = session.checklist
    if fields == ["rooms"] and _ADD_WORDS.search(clean.lower()) and c.rooms:
        # "add one room" -> rooms + n, keeping the room types already chosen
        nums = _numbers_in(clean) - {0}
        n = next(iter(nums)) if len(nums) == 1 else 1
        session.update_checklist(rooms=min(c.rooms + n, 10))
        session.readback_done = False
        session.awaiting_change = False
        session.set_stage(Stage.COLLECTING_INFO)
    else:
        _reset_fields(session, fields)
    return await _next_question(session)


async def _handle_final(session: Session, clean: str) -> str:
    if not session.readback_done:
        return _final_readback(session)
    fields = _fields_to_change(clean)
    if session.awaiting_change:
        if not fields:
            return "Sorry, " + _WHICH_DETAIL[0].lower() + _WHICH_DETAIL[1:]
        return await _apply_change(session, clean, fields)
    ans = _yes_no(clean)
    wants_change = bool(fields) and (ans != "yes" or bool(_CHANGE_WORDS.search(clean.lower())))
    if wants_change:
        return await _apply_change(session, clean, fields)
    if ans == "yes":
        return await _create_booking_now(session)
    if ans in ("no", "correction"):
        session.awaiting_change = True
        return "No problem. " + _WHICH_DETAIL
    return "Sorry, I didn't catch that. Shall I confirm this booking - yes or no?"

# ----------------------------------------------------------------------
# Tool plumbing
# ----------------------------------------------------------------------
_FIELD_TOOL_MAP: dict[str, set[str]] = {
    "city": {"list_hotels"},
    "hotel": {"get_room_options", "confirm_field", "list_hotels"},
    "first_name": {"record_field_attempt", "confirm_field"},
    "last_name": {"record_field_attempt", "confirm_field"},
    "check_in": {"record_field_attempt", "confirm_field"},
    "check_out": {"record_field_attempt", "confirm_field"},
    "email": {"record_field_attempt", "confirm_field"},
    "phone": {"confirm_contact_number"},
    "phone_confirmed": {"confirm_contact_number"},
    "party": {"validate_room_count"},
    "room_type": {"get_room_options", "confirm_field"},
}
_ALWAYS_ALLOWED_TOOLS = {"list_cities"}


def _tools_for_next_step(session: Session) -> list[dict[str, Any]]:
    step = _current_step(session)
    allowed = set(_FIELD_TOOL_MAP.get(step, set())) | _ALWAYS_ALLOWED_TOOLS
    return [t for t in TOOL_DEFINITIONS if t["function"]["name"] in allowed]


@dataclass
class _Turn:
    """Per-customer-message scratchpad (a fresh one for every message)."""
    user_text: str
    fields_recorded: set = field(default_factory=set)
    tools_used: set = field(default_factory=set)
    pending_set: bool = False
    forced: tuple | None = None
    invalid_field: str | None = None
    invalid_msg: str = ""
    phone_invalid: bool = False
    party_missing: bool = False
    rooms_before: int = 0
    party_shortfall: str | None = None
    info_tool: bool = False
    blocked: bool = False


def _register_guess(session: Session, field_name: str, value: str, turn: _Turn) -> None:
    """A valid-looking name/email/date guess: park it as pending for a
    yes/no read-back - or, after too many tries, accept it."""
    if field_name in _RETRY_FIELDS:
        n = session.field_attempts.get(field_name, 0) + 1
        session.field_attempts[field_name] = n
        if n >= _MAX_FIELD_ATTEMPTS:
            session.update_checklist(**{field_name: value})
            session.locked_fields.add(field_name)
            session.pending = None
            turn.forced = (field_name, value)
            return
    session.pending = {"field": field_name, "value": value}
    turn.pending_set = True


def _register_invalid(session: Session, field_name: str, msg: str, turn: _Turn) -> None:
    key = f"{field_name}:invalid"
    session.field_attempts[key] = session.field_attempts.get(key, 0) + 1
    turn.invalid_field = field_name
    turn.invalid_msg = msg


async def _run_tool_calls(session: Session, tool_calls: list[Any], turn: _Turn) -> list[dict[str, Any]]:
    results = []
    c = session.checklist
    for call in tool_calls:
        name = call.function.name
        try:
            args = json.loads(call.function.arguments)
        except (json.JSONDecodeError, TypeError):
            args = {}

        # Never trust the model's hotel/city strings once they're locked.
        if name == "get_room_options":
            if c.hotel:
                args["hotel"] = c.hotel
            if c.city:
                args["city"] = c.city

        def used_twice() -> str:
            return json.dumps({"already_used_this_turn": True,
                               "instruction": "You already called this for the customer's last message. Stop calling tools."})

        content: str
        if name == "create_booking":
            content = json.dumps({"not_allowed": True,
                                  "instruction": "The system creates the booking itself. Do not call this."})

        elif name == "get_room_options" and getattr(session, "available_room_types", None):
            matched = next((t for t in session.available_room_types
                            if t.lower() in turn.user_text.lower()), None)
            if matched and _current_step(session) == "room_type":
                output = await EXECUTORS["confirm_field"](field="room_type", value=matched)
                await _apply_tool_result(session, "confirm_field", {"field": "room_type", "value": matched}, output)
                content = json.dumps(output)
            else:
                output = {"room_types": [{"type": t, "price_per_night": session.room_prices.get(t)}
                                         for t in session.available_room_types]}
                content = json.dumps(output)
                if _current_step(session) == "room_type":
                    turn.info_tool = True

        elif name == "list_cities" and getattr(session, "available_cities", None):
            matched_city = next((x for x in session.available_cities if x.lower() in turn.user_text.lower()), None)
            if matched_city and _current_step(session) == "city":
                output = await EXECUTORS["list_hotels"](city=matched_city)
                await _apply_tool_result(session, "list_hotels", {"city": matched_city}, output)
            else:
                output = {"cities": session.available_cities}
                turn.info_tool = _current_step(session) == "city"
            content = json.dumps(output)

        elif name == "record_field_attempt":
            field_name = args.get("field", "")
            expected = _current_step(session)
            if field_name != expected:
                content = json.dumps({"error": "wrong_field", "expected_field": expected,
                                      "instruction": f"You are asking about {expected}, not {field_name}."})
            elif field_name in session.locked_fields:
                content = json.dumps({"error": "already_confirmed",
                                      "confirmed_value": getattr(c, field_name, None)})
            elif field_name in turn.fields_recorded:
                content = json.dumps({"error": "already_attempted_this_turn",
                                      "instruction": "Stop calling tools; the system asks the customer."})
            else:
                turn.fields_recorded.add(field_name)
                value, ok, msg = _validate_field(session, field_name, str(args.get("value", "")))
                if ok:
                    _register_guess(session, field_name, value, turn)
                    content = json.dumps({"recorded": True, "value": value})
                else:
                    _register_invalid(session, field_name, msg, turn)
                    content = json.dumps({"valid": False, "instruction": "Not usable. Stop; the system asks again."})

        elif name == "confirm_field":
            field_name = args.get("field", "")
            value = str(args.get("value", ""))
            expected = _current_step(session)
            if name in turn.tools_used:
                content = used_twice()
            else:
                turn.tools_used.add(name)
                if field_name != expected:
                    content = json.dumps({"error": "wrong_field", "expected_field": expected,
                                          "instruction": f"You are on {expected}, not {field_name}."})
                elif field_name in _CONFIRM_FIELDS:
                    # Only a "yes" from the customer (checked in code) locks these.
                    content = json.dumps({"not_confirmed": True,
                                          "instruction": "Names, email and dates are confirmed by the system. Stop."})
                elif field_name == "hotel":
                    match = _match_option(value, session.available_hotels)
                    if match is None:
                        content = json.dumps({"error": "hotel_not_recognized",
                                              "valid_options": session.available_hotels})
                    else:
                        output = await EXECUTORS["confirm_field"](field="hotel", value=match)
                        await _apply_tool_result(session, name, {"field": "hotel", "value": match}, output)
                        content = json.dumps(output)
                elif field_name == "room_type":
                    match = _match_option(value, session.available_room_types)
                    if match is None:
                        content = json.dumps({"error": "room_type_not_recognized",
                                              "valid_options": session.available_room_types})
                    else:
                        output = await EXECUTORS["confirm_field"](field="room_type", value=match)
                        await _apply_tool_result(session, name, {"field": "room_type", "value": match}, output)
                        content = json.dumps(output)
                else:
                    output = await EXECUTORS["confirm_field"](**args)
                    await _apply_tool_result(session, name, args, output)
                    content = json.dumps(output)

        elif name == "confirm_contact_number":
            if name in turn.tools_used:
                content = used_twice()
            else:
                turn.tools_used.add(name)
                digits = re.sub(r"\D", "", str(args.get("number", "")))
                if len(digits) == 12 and digits.startswith("91"):
                    digits = digits[2:]
                if len(digits) == 11 and digits.startswith("0"):
                    digits = digits[1:]
                if not (10 <= len(digits) <= 13):
                    turn.phone_invalid = True
                    content = json.dumps({"invalid_number": True,
                                          "instruction": "Not a usable phone number. Stop; the system asks again."})
                else:
                    output = await EXECUTORS["confirm_contact_number"](number=digits, confirmed_caller_id=True)
                    await _apply_tool_result(session, name, {"number": digits}, output)
                    content = json.dumps(output)

        elif name == "validate_room_count":
            if name in turn.tools_used:
                content = used_twice()
            else:
                turn.tools_used.add(name)
                try:
                    a, ch, r = int(args["adults"]), int(args["children"]), int(args["rooms"])
                except (KeyError, TypeError, ValueError):
                    a = ch = r = None
                said = _numbers_in(turn.user_text)
                if a is None or not {a, ch, r} <= said:
                    # The customer did not actually say these numbers.
                    turn.party_missing = True
                    content = json.dumps({"numbers_not_given": True,
                                          "instruction": "The customer has not said these numbers. Stop; the system asks."})
                else:
                    output = await EXECUTORS["validate_room_count"](adults=a, children=ch, rooms=r)
                    await _apply_tool_result(session, name, {"adults": a, "children": ch, "rooms": r}, output)
                    if not output.get("valid"):
                        turn.party_shortfall = (
                            f"For {_plural(a, 'adult', 'adults')} and {_plural(ch, 'child', 'children')} "
                            f"you need at least {_plural(output['minimum_rooms_required'], 'room', 'rooms')}.")
                    content = json.dumps(output)

        else:
            executor = EXECUTORS.get(name)
            if executor is None:
                content = json.dumps({"error": f"unknown tool '{name}'"})
            else:
                try:
                    t0 = time.perf_counter()
                    output = await executor(**args)
                    print(f"  [timing] tool {name}: {time.perf_counter() - t0:.2f}s")
                    await _apply_tool_result(session, name, args, output)
                    content = json.dumps(output)
                    if name in ("list_cities", "list_hotels") and "error" not in output:
                        turn.info_tool = True
                except Exception as exc:
                    content = json.dumps({"error": f"tool call failed: {exc}"})

        if '"error"' in content:
            turn.blocked = True
        results.append({"role": "tool", "tool_call_id": call.id, "name": name, "content": content})
    return results


# ----------------------------------------------------------------------
# Reply selection
# ----------------------------------------------------------------------
def _reply_fits_step(session: Session, reply: str, step: str) -> bool:
    low = reply.lower()
    if any(k in low for k in _STEP_KEYWORDS.get(step, ())):
        return True
    if step == "city" and _mentions_any(reply, session.available_cities or []):
        return True
    if step == "hotel" and _mentions_any(reply, session.available_hotels or []):
        return True
    if step == "room_type" and _mentions_any(reply, session.available_room_types or []):
        return True
    return False


def _skips_ahead(reply: str, step: str) -> bool:
    if step not in _PERSONAL_STEPS:
        return False
    low = reply.lower()
    idx = _FIELD_ORDER.index(step)
    for later in _FIELD_ORDER[idx + 1:]:
        if any(k in low for k in _LATER_KEYWORDS.get(later, ())):
            return True
    return False


async def _reply_from_turn(session: Session, turn: _Turn, step_before: str) -> str | None:
    """Turns what the tools did this round into the (code-written) reply."""
    if turn.pending_set and session.pending:
        return _readback(session.pending["field"], session.pending["value"])
    if turn.forced:
        _, value = turn.forced
        return (f"I'm having trouble catching that clearly, so I've noted it as {value} "
                f"and we'll double check it before your stay. " + await _next_question(session))
    if turn.invalid_field:
        return _junk_reply(session, turn.invalid_field, turn.invalid_msg)
    if turn.phone_invalid:
        return "Sorry, I didn't catch that number. " + _STEP_QUESTIONS["phone"]
    if turn.party_missing:
        return "Sorry, I didn't catch that. " + _STEP_QUESTIONS["party"]
    if turn.party_shortfall:
        return turn.party_shortfall + " How many adults, children and rooms would you like?"
    if _current_step(session) != step_before:
        return await _next_question(session)
    if len(session.checklist.room_type or []) != turn.rooms_before:
        return await _next_question(session)      # next room's type
    if turn.info_tool:
        return await _next_question(session)
    if turn.blocked:
        return "Sorry, I didn't catch that. " + await _next_question(session)
    return None


async def _fallback_for_step(session: Session, step: str, clean: str, turn: _Turn) -> str:
    """The model answered in plain text at a name/email/date step without
    recording anything. Try to read the value ourselves; else re-ask."""
    value, ok, msg = _validate_field(session, step, clean, strict=True)
    if ok:
        _register_guess(session, step, value, turn)
        reply = await _reply_from_turn(session, turn, step)
        if reply:
            return reply
    _register_invalid(session, step, msg, turn)
    return _junk_reply(session, step, msg)

async def _handle_counts(session: Session, step: str, clean: str) -> str:
    """Adults / children / rooms are read in code - no model involved."""
    c = session.checklist
    if step == "adults":
        a = _count_for(clean, _ADULT_NOUNS)
        if a is None:
            a = _single_number(clean)
        if a is None or not 1 <= a <= 20:
            return "Sorry, I didn't catch that. " + _STEP_QUESTIONS["adults"]
        session.update_checklist(adults=a)
        ch = _count_for(clean, _CHILD_NOUNS)          # "2 adults and 1 child"
        if ch is not None and 0 <= ch <= 20:
            session.update_checklist(children=ch)
    elif step == "children":
        ch = _count_for(clean, _CHILD_NOUNS)
        if ch is None:
            ch = _single_number(clean)
        if ch is None or not 0 <= ch <= 20:
            return "Sorry, I didn't catch that. " + _STEP_QUESTIONS["children"]
        session.update_checklist(children=ch)
    else:
        r = _count_for(clean, _ROOM_NOUNS)
        if r is None:
            r = _single_number(clean)
        if r is None or not 1 <= r <= 10:
            return "Sorry, I didn't catch that. " + _step_question(session, "rooms")
        if r < _min_rooms(c.adults, c.children):
            return _step_question(session, "rooms")      # states the minimum again
        session.update_checklist(rooms=r)
        if c.room_type:
            c.room_type = c.room_type[:r]                # fewer rooms: drop extras
    return await _next_question(session)


async def _take_room_types(session: Session, clean: str) -> bool:
    """Room type(s) named exactly in the answer, e.g. 'Suite', 'Deluxe for
    both', or two names for two rooms. Fuzzy mishearings go to the model."""
    await _prefetch(session, "room_type")
    low = clean.lower()
    hits = sorted((low.find(t.lower()), t) for t in session.available_room_types or []
                  if t.lower() in low)
    remaining = (session.checklist.rooms or 1) - len(session.checklist.room_type or [])
    fill_all = bool(re.search(r"\b(both|all|same|each|every)\b", low)) and len(hits) == 1
    if not hits or (len(hits) > remaining and not fill_all):
        return False
    _add_room_types(session, [t for _, t in hits], fill_all=fill_all)
    return True

# ----------------------------------------------------------------------
# Hotel knowledge base (OKF) - answers caller QUESTIONS in code
# ----------------------------------------------------------------------
# The booking flow above never answered questions about the hotel (any
# off-step reply is replaced by the next booking question). When the caller
# asks something the knowledge base can answer (facilities, dining, address,
# cancellation policy, room details ...) we answer it from the Markdown files
# generated from MongoDB (okf_store) and then repeat the question they were
# in the middle of, so the booking flow carries on exactly where it was.
# No model call, no state change: only read-only lookups.
_Q_START = re.compile(
    r"^(?:(?:and|also|ok|okay|so|hi|hello|please|just|one more thing|by the way)\s+)*"
    r"(?:do|does|did|is|are|was|were|can|could|will|would|what|what's|whats|which|"
    r"where|where's|when|how|tell me|any|is there|are there|may i know|"
    r"i want to know|i wanted to know|i would like to know|can you tell)\b",
    re.I)
_KB_COUNT_STEPS = {"adults", "children", "rooms", "room_type"}
_KB_NEEDS_HOTEL = ("That depends on the hotel, so I'll share the details "
                   "once you've chosen one.")
_KB_POLICY_VARIES = ("Those terms depend on the room type, so I'll go through "
                     "them once you've picked a room.")
_KB_MAX_CHARS = 320


def _is_info_question(text: str) -> bool:
    return "?" in text or bool(_Q_START.search(text.strip()))


def _clip_spoken(text: str, limit: int = _KB_MAX_CHARS) -> str:
    """Keeps whole sentences up to ~limit characters (it is read aloud)."""
    text = " ".join(text.split())
    if len(text) > limit:
        out = ""
        for part in re.split(r"(?<=[.!?])\s+", text):
            if out and len(out) + 1 + len(part) > limit:
                break
            out = (out + " " + part).strip()
        text = out if len(out) <= limit else out[:limit].rsplit(" ", 1)[0].rstrip(",;:")
    return text if text[-1:] in ".!?" else text + "."


def _pick_kb_snippet(session: Session, snips: list) -> Any:
    """Best snippet, or _KB_POLICY_VARIES when cancellation/payment terms
    differ per room type and the caller hasn't chosen a room yet."""
    if not snips:
        return None
    top = snips[0]
    if top.type == "policy" and top.meta.get("room_id"):
        pol = [s for s in snips if s.type == "policy" and s.meta.get("room_id")]
        chosen = [t.lower() for t in (session.checklist.room_type or [])]
        for s in pol:
            if s.title.lower().removesuffix(" policies") in chosen:
                return s
        if len(pol) > 1 and pol[1].score == top.score and pol[1].spoken != top.spoken:
            return _KB_POLICY_VARIES
    return top


async def _reask_current(session: Session, step: str) -> str:
    """What the agent would say next - without changing any booking state."""
    if step == "_done":
        if session.awaiting_change:
            return _WHICH_DETAIL
        if session.readback_done:
            return "Shall I confirm this booking - yes or no?"
        return _final_readback(session)
    if step in ("phone", "phone_confirmed"):
        return _phone_question(session)
    pend = session.pending
    if pend and pend["field"] == step:
        return _readback(pend["field"], pend["value"])      # still waiting for their yes/no
    return await _next_question(session)


async def _kb_answer(session: Session, user_message: str, clean: str, step: str) -> str | None:
    """A reply built from the knowledge base, or None to let the normal
    booking flow handle the message (the default for anything uncertain)."""
    if not okf_store.ready():
        return None
    if user_message.lstrip().startswith("[transcription uncertain"):
        return None                       # don't answer a question we may have misheard
    if not _is_info_question(clean):
        return None                       # a booking answer ("John", "5th March") - not a question
    topics = okf_store.topics_in(clean)
    if step in _KB_COUNT_STEPS:
        if _numbers_in(clean):
            return None                   # "2 rooms please" etc. is an answer
        topics = topics - {"room"}        # "do you have a suite?" belongs to the booking step
    if not topics:
        return None

    named = okf_store.hotel_in(clean)              # hotel the caller mentioned, if any
    hotel = named or session.checklist.hotel
    lead = ""
    if hotel:
        pick = _pick_kb_snippet(session, okf_store.lookup(clean, hotel=hotel, limit=3,
                                                          require_topic=True))
        if pick is None:
            return None
        answer = pick if isinstance(pick, str) else pick.spoken
        if not isinstance(pick, str):
            print(f"  [kb] answered from {pick.path} (score {pick.score:.0f})")
            if not named and pick.type != "hotel_overview":
                # say WHICH hotel this is about (the caller may have meant another)
                lead = f"Here's what I have for {session.checklist.hotel}. "
        if not answer:
            return None
    else:
        if topics <= {"overview"} or not okf_store.lookup(clean, limit=1, require_topic=True):
            return None                   # nothing to say yet / nothing known: normal flow
        answer = _KB_NEEDS_HOTEL
    return lead + _clip_spoken(answer) + " " + await _reask_current(session, step)

# ----------------------------------------------------------------------
# Hotel options menu - offered once, right after the caller picks a hotel
#   1 Book this hotel   2 Hotel amenities   3 Restaurants
#   4 Nearby places     5 Ask a question
# Every answer is read from that hotel's OKF files (okf_store). If the
# files don't hold something, the caller is told so - nothing is invented.
# "Book this hotel" hands over to the normal booking steps (first name next).
# Switch off with HOTEL_MENU=off in .env (the flow then behaves as before).
# ----------------------------------------------------------------------
_MENU_NEXT = "Anything else, or shall I book this hotel?"
_ITEMS_PER_TURN = 8          # list items read per turn ("more" reads the next ones)
_TEXT_CHARS = 330            # characters read per turn for descriptions

_BOOK_RE = re.compile(
    r"\b(book (this|it|a room|now|here|the (hotel|room|stay))|let'?s book|"
    r"(want|like|wanna|need) to book|go ahead and book|(can|could|may) i (book|reserve)|"
    r"proceed (with|to) (the )?(booking|reservation)|reserve (this|it|a room)|"
    r"make a (booking|reservation))\b")
_NOT_ROOM_RE = re.compile(          # "book a TABLE / the SPA / a CAB" is not booking the hotel
    r"\b(table|spa|massage|banquet|hall|cab|taxi|tour|ticket|flight|event|wedding|conference|meeting)\b")
_BOOK_ONLY_RE = re.compile(r"^(?:(?:yes|ok|okay|please) )*(?:book|booking|reserve)(?: please)?$")
_MENU_RE = re.compile(r"\b(main menu|the options|my options|what (else )?can you (do|help)|go back|start over)\b")
_CHANGE_HOTEL_RE = re.compile(r"\b(another|different|other|change( the)?) (hotel|property)\b")
_NO_MORE_RE = re.compile(r"\b(nothing|that'?s all|that is all|no thanks|no thank you)\b")
_MORE_RE = re.compile(
    r"^(?:(?:yes|ok|okay|please|and) )*(?:more|go on|continue|keep going|the rest|next|read more|tell me more)(?: please)?$")
_CHOICE_RE = re.compile(
    r"^(?:option |number |choice )?(1|2|3|4|5|one|two|three|four|five|first|second|third|fourth|fifth)(?: one| please| option)?$")
_CHOICE_WORDS = {"1": "book", "one": "book", "first": "book",
                 "2": "amenities", "two": "amenities", "second": "amenities",
                 "3": "restaurants", "three": "restaurants", "third": "restaurants",
                 "4": "nearby", "four": "nearby", "fourth": "nearby",
                 "5": "ask", "five": "ask", "fifth": "ask"}
_SECTION_RE = {
    "amenities": re.compile(r"\b(amenit\w*|facilit\w*|features?|services?)\b"),
    "restaurants": re.compile(r"\b(restaurants?|dining|dine|food|eat|eating|cafe|cuisines?)\b"),
    "nearby": re.compile(r"\b(nearby|near by|near|attractions?|sightseeing|places?|around|landmarks?|tourist\w*)\b"),
    "ask": re.compile(r"\b(questions?|ask|enquir\w*|inquir\w*|information|info|something else)\b"),
}
_DIST_Q = re.compile(r"\b(how far|distance|far|km|kms|kilomet\w*|miles?|minutes?|mins?|drive|driving|walk\w*|away|how long)\b")
_TEXT_DOCS = ("nearby-attractions.md", "dining.md", "wellness.md",
              "events-and-banquets.md", "experiences.md")
# an amenity (or the question) -> the hotel's own description file for it
_AMENITY_SECTIONS = (
    (("spa", "wellness", "massage", "sauna", "steam"), "amenities/wellness.md", "wellness"),
    (("restaurant", "dining", "bar", "food", "cafe", "buffet"), "amenities/dining.md", "dining"),
    (("banquet", "hall", "event", "conference", "wedding", "meeting"),
     "amenities/events-and-banquets.md", "events and banquets"),
    (("experience", "activity", "activities", "excursion", "tour"),
     "amenities/experiences.md", "activities and experiences"),
)
_GENERIC = {
    "the", "is", "are", "a", "an", "do", "you", "have", "what", "which", "of", "to", "in", "on", "at",
    "for", "and", "or", "me", "my", "i", "can", "does", "there", "it", "with", "tell", "about", "more",
    "please", "hotel", "any", "how", "where", "when", "yes", "no", "detail", "details", "info",
    "information", "available", "offer", "offers", "give", "show", "want", "know", "like", "would",
    "could", "this", "that", "your", "their", "some", "also", "much", "many", "get", "there's"}


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def _toks(text: str) -> set[str]:
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) > 1 and t not in _GENERIC}


_TOPIC_WORDS = {_stem(w) for w in (
    "restaurant", "restaurants", "dining", "dine", "food", "eat", "eating", "cafe", "nearby", "near",
    "place", "places", "attraction", "attractions", "sightseeing", "around", "far", "distance", "away",
    "amenity", "amenities", "facility", "facilities", "service", "services", "km", "kms", "minutes",
    "minute", "mins", "drive", "walk", "long", "nearest", "closest")}
# words that point at a stored label of a restaurant / place row
_LABEL_HINTS = {
    "timings": {_stem(w) for w in ("time", "timing", "timings", "open", "opening", "close", "closing", "hour", "hours")},
    "cuisine": {_stem(w) for w in ("cuisine", "cuisines", "serve", "serves", "dish", "dishes")},
}


def _menu_pending(session: Session) -> bool:
    """True from the moment a hotel is chosen until the caller picks
    'Book this hotel'. Off when HOTEL_MENU=off or the knowledge base isn't
    loaded (then the booking flow carries on exactly as before)."""
    if os.getenv("HOTEL_MENU", "on").strip().lower() in ("off", "0", "false", "no"):
        return False
    return bool(session.checklist.hotel) and not session.hotel_menu["done"] and okf_store.ready()


def _menu_prompt(session: Session, again: bool = False, lead: str | None = None) -> str:
    if session.stage == Stage.INFO:
        return (lead if lead is not None else "Sure. ") + _INFO_OPTIONS
    if session.stage == Stage.POST_BOOKING:
        return (lead if lead is not None else ("Sure. " if again else "")) + _POST_OPTIONS
    if lead is None:
        lead = "Sure. " if again else f"Great choice - {session.checklist.hotel}. "
    return (lead + "What would you like to do? You can book this hotel, hear about its "
            "amenities, its restaurants or nearby places, or ask me a question.")


# ---- reading the hotel's OKF files ------------------------------------
def _hotel_doc(session: Session, path_end: str):
    for d in okf_store.hotel_docs(session.checklist.hotel):
        if d.path.replace("\\", "/").endswith(path_end):
            return d
    return None


def _doc_facts(doc) -> list[tuple[str, str]]:
    """The '- Label: value' lines of a file's Details section."""
    out: list[list[str]] = []
    for line in (getattr(doc, "details", "") or "").splitlines():
        m = re.match(r"^\s*-\s*([^:]+):\s*(.*)$", line)
        if m:
            out.append([m.group(1).strip(), m.group(2).strip()])
        elif out and line.strip():
            out[-1][1] += " " + line.strip()
    return [(k, v) for k, v in out if v]


def _fact_text(doc) -> str:
    facts = _doc_facts(doc)
    return facts[0][1] if facts else ""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _amenity_items(session: Session) -> list[str]:
    """The hotel's amenities, from the `amenities` collection (amenity-list.md)."""
    items: list[str] = []
    for _label, value in _doc_facts(_hotel_doc(session, "amenities/amenity-list.md")):
        items += [x.strip() for x in value.split(", ") if x.strip()]
    seen: set[str] = set()
    return [i for i in items if not (i.lower() in seen or seen.add(i.lower()))]


def _doc_items(doc) -> list[dict]:
    """Entries of a list file (restaurants.md / nearby-places.md):
    '- Restaurant: Name' followed by indented '  - Label: value' lines."""
    items: list[dict] = []
    for line in (getattr(doc, "details", "") or "").splitlines():
        m = re.match(r"^(\s*)-\s*([^:]+):\s*(.*)$", line)
        if not m:
            continue
        label, value = m.group(2).strip(), m.group(3).strip()
        if not m.group(1):
            items.append({"name": value, "facts": []})
        elif items and value:
            items[-1]["facts"].append((label, value))
    return [i for i in items if i["name"]]


def _named_items(items: list[dict], clean: str) -> list[dict]:
    """The row(s) the caller named, e.g. 'tell me about Spice Route'."""
    q = _toks(clean) - _TOPIC_WORDS
    scored = []
    for it in items:
        t = _toks(it["name"])
        n = len(t & q)
        if t and n and n / len(t) >= 0.5:
            scored.append((n, it))
    top = max((n for n, _ in scored), default=0)
    return [it for n, it in scored if n == top]


def _distance_of(it: dict) -> str | None:
    return next((v for k, v in it["facts"] if k == "Distance"), None)


def _item_line(it: dict) -> str:
    """Everything stored for one row, spoken: name, then 'Label: value.'"""
    d = _distance_of(it)
    head = f"{it['name']} is {d} away." if d else f"{it['name']}."
    rest = " ".join(f"{k}: {v.rstrip('.')}." for k, v in it["facts"] if k != "Distance")
    return f"{head} {rest}".strip()


# ---- speaking long answers a bit at a time ----------------------------
def _take_units(units: list[str], kind: str) -> tuple[list[str], list[str]]:
    if kind == "items":
        return units[:_ITEMS_PER_TURN], units[_ITEMS_PER_TURN:]
    out: list[str] = []
    used = 0
    for i, u in enumerate(units):
        if out and used + len(u) > _TEXT_CHARS:
            return out, units[i:]
        out.append(u)
        used += len(u) + 1
    return out, []


def _speak_units(session: Session, units: list[str], kind: str, lead: str = "", tail: str = "") -> str:
    head, rest = _take_units(units, kind)
    session.hotel_menu["queue"], session.hotel_menu["kind"] = rest, kind
    text = lead + (_spoken_list(head) + "." if kind == "items" else " ".join(head))
    if rest:
        text += (f" And {len(rest)} more - say more to hear them." if kind == "items"
                 else " There's more - say more to hear it.")
    else:
        text += (" " + tail if tail else "") + " " + _MENU_NEXT
    return " ".join(text.split())


def _say_more(session: Session) -> str:
    m = session.hotel_menu
    if not m["queue"]:
        return "That's everything I have on this. " + _MENU_NEXT
    return _speak_units(session, m["queue"], m["kind"])


def _best_sentences(text: str, clean: str) -> list[str]:
    q = _toks(clean) - _TOPIC_WORDS
    if not q:
        return []
    scored = [(len(_toks(s) & q), i, s) for i, s in enumerate(_sentences(text))]
    scored = [x for x in scored if x[0]]
    if not scored:
        return []
    top = max(x[0] for x in scored)
    return [s for _, _, s in sorted((x for x in scored if x[0] == top), key=lambda x: x[1])[:2]]


def _only_topic_words(clean: str) -> bool:
    return not (_toks(clean) - _TOPIC_WORDS)


# ---- 2. Hotel amenities ------------------------------------------------
def _amenities_intro(session: Session) -> str:
    hotel = session.checklist.hotel
    items = _amenity_items(session)
    if not items:
        return f"I don't have an amenities list for {hotel}. " + _MENU_NEXT
    return _speak_units(session, items, "items", lead=f"{hotel} offers ",
                        tail="Ask me about any of them for details.")


def _amenity_followup(session: Session, clean: str) -> str | None:
    q = _toks(clean)
    best, best_n = None, 0
    for it in _amenity_items(session):
        t = _toks(it)
        n = len(t & q)
        if t and n and n / len(t) >= 0.5 and n > best_n:
            best, best_n = it, n
    words = q | (_toks(best) if best else set())
    for keys, path, label in _AMENITY_SECTIONS:
        if {_stem(k) for k in keys} & words:
            text = _fact_text(_hotel_doc(session, path))
            if text:
                lead = (f"Here's what the hotel says about its {label}. "
                        if not best or best[0].isdigit() else
                        f"{best[0].upper()}{best[1:]} - here's what the hotel says about its {label}. ")
                return _speak_units(session, _sentences(text), "text", lead=lead)
    if best:
        return (f"{best[0].upper()}{best[1:]} is listed among the facilities at "
                f"{session.checklist.hotel}, but I don't have a further description of it. " + _MENU_NEXT)
    return None


# ---- 3. Restaurants (the `restaurants` collection -> restaurants.md) ----
def _restaurant_items(session: Session) -> list[dict]:
    return _doc_items(_hotel_doc(session, "amenities/restaurants.md"))


def _restaurants_intro(session: Session) -> str:
    hotel = session.checklist.hotel
    items = _restaurant_items(session)
    if not items:
        return f"I don't have restaurant details on record for {hotel}. " + _MENU_NEXT
    n = len(items)
    return _speak_units(session, [i["name"] for i in items], "items",
                        lead=f"{hotel} has {n} restaurant{'s' if n != 1 else ''}: ",
                        tail="Ask me about any of them.")


def _restaurants_followup(session: Session, clean: str) -> str | None:
    items = _restaurant_items(session)
    if not items:
        return None
    named = _named_items(items, clean)
    q = _toks(clean)
    if named:                                              # "tell me about Spice Route"
        lines = []
        for it in named:
            have = {k.lower() for k, _ in it["facts"]}
            for hint, words in _LABEL_HINTS.items():       # asked for something it doesn't have
                if words & q and hint not in have:
                    lines.append(f"I don't have the {hint} for {it['name']}.")
            lines.append(_item_line(it))
        return _speak_units(session, lines, "text")
    lines = []                                             # "what cuisine" / "what time open"
    for it in items:
        for k, v in it["facts"]:
            if (_toks(k) & q) or (_LABEL_HINTS.get(k.lower(), set()) & q):
                lines.append(f"{k}: {v.rstrip('.')}." if len(items) == 1
                             else f"{it['name']} - {k.lower()}: {v.rstrip('.')}.")
    return _speak_units(session, lines, "text") if lines else None


# ---- 4. Nearby places (the `nearbyhotels` collection -> nearby-places.md)
def _nearby_items(session: Session) -> list[dict]:
    return _doc_items(_hotel_doc(session, "directions/nearby-places.md"))


def _nearby_intro(session: Session) -> str:
    hotel = session.checklist.hotel
    items = _nearby_items(session)
    if not items:
        return f"I don't have nearby places on record for {hotel}. " + _MENU_NEXT
    tail = "Ask me how far any of them is." if any(_distance_of(i) for i in items) else ""
    return _speak_units(session, [i["name"] for i in items], "items",
                        lead="Nearby places include ", tail=tail)


def _nearby_followup(session: Session, clean: str) -> str | None:
    items = _nearby_items(session)
    if not items:
        return None
    named = _named_items(items, clean)
    if _DIST_Q.search(clean.lower()):                      # "how far is X"
        if named:
            pool = named
        elif _toks(clean) - _TOPIC_WORDS:                  # a place that is not in the records
            return "I don't have that place in the hotel's records. " + _MENU_NEXT
        else:                                              # "how far are the places"
            pool = items
        lines = [f"{i['name']} is {_distance_of(i)} away." for i in pool if _distance_of(i)]
        if lines:
            return _speak_units(session, lines, "text")
        return "The hotel's records don't give a distance for that. " + _MENU_NEXT
    if named:
        return _speak_units(session, [_item_line(i) for i in named], "text")
    return None


# ---- 5. Ask a question -------------------------------------------------
def _kb_lookup(session: Session, clean: str) -> str | None:
    """Best answer for a free question about THIS hotel, from the KB."""
    hotel = session.checklist.hotel
    if okf_store.topics_in(clean) == {"overview"}:        # "tell me about the helipad" is NOT
        left = _toks(re.sub(r"tell me about|about (the|this) hotel|overview", " ", clean.lower()))
        if left - _toks(hotel):                            # ... the hotel overview
            return _search_hotel_text(hotel, clean)
    named = okf_store.topics_in(clean) - {"overview"}
    snips = okf_store.lookup(clean, hotel=hotel, limit=3, require_topic=True)
    if not snips:
            snips = [s for s in okf_store.lookup(clean, hotel=hotel, limit=3)
                 if s.score >= 4 and s.type != "hotel_overview"]
    if named:                                              # asked about a specific topic:
        snips = [s for s in snips if s.type != "hotel_overview"]   # never answer with the overview
    pick = _pick_kb_snippet(session, snips)
    if pick is None:
        return _search_hotel_text(hotel, clean)
    if isinstance(pick, str):
        return pick
    print(f"  [menu] answered from {pick.path} (score {pick.score:.0f})")
    if pick.path.replace("\\", "/").endswith(_TEXT_DOCS):
        return _fact_text(pick) or pick.spoken or None      # the hotel's full text
    return pick.spoken or None

_RATING_RE = re.compile(r"\b(ratings?|stars?)\b")

async def _catalogue_answer(session: Session, clean: str) -> str | None:
    """'Other hotels in Ahmedabad' / 'which hotels do you have' / 'which cities':
    answered from Mongo (read-only), because the knowledge files are per hotel and
    never list other hotels."""
    low = clean.lower()
    if not re.search(r"\b(hotels|cities)\b", low) or re.search(r"\b(nearby|near by|near)\b", low):
        return None
    if _RATING_RE.search(low):          # "five star hotels" is a rating question, not a city list
        return None
    await _prefetch(session, "city")                       # loads session.available_cities
    cities = session.available_cities or []
    if re.search(r"\bcities\b", low) and not re.search(r"\bhotels\b", low):
        return f"We have hotels in {_spoken_list(cities)}."
    city = next((c for c in cities if c.lower() in low), None) or session.checklist.city
    if not city:
        return ("Which city would you like to know the hotels for? "
                f"We have hotels in {_spoken_list(cities)}.")
    out = await EXECUTORS["list_hotels"](city=city)        # pure read
    names = out.get("hotels") or []
    if not names:
        return f"I don't have any hotels on record in {city}."
    current = session.checklist.hotel
    if current and re.search(r"\b(other|else|another|more)\b", low):
        names = [n for n in names if n.lower() != current.lower()]
        if not names:
            return f"{current} is the only hotel we have in {city}."
        return f"The other hotels in {city} are {_spoken_list(names)}."
    return f"In {city} we have {_spoken_list(names)}."

def _search_hotel_text(hotel: str | None, clean: str) -> str | None:
    """Last resort: the stored lines (any file of this hotel) that best match the caller's
    words. Reads only what the knowledge files contain - nothing is added."""
    if not hotel:
        return None
    q = _toks(clean) - _toks(hotel) - {"other", "another", "else", "anything"}
    if not q:
        return None
    need = min(2, len(q))
    hits: list[tuple[int, int, str]] = []
    order = 0
    for d in okf_store.hotel_docs(hotel):
        parent = ""
        for raw in d.details.splitlines():
            m = re.match(r"^(\s*)-\s*([^:]+):\s*(.*)$", raw)
            if not m or not m.group(3).strip():
                continue
            indent, label, value = m.group(1), m.group(2).strip(), m.group(3).strip()
            if not indent:
                parent = value if label in ("Restaurant", "Place") else ""
            line = f"{parent}, {label.lower()}: {value}" if indent and parent else f"{label}: {value}"
            order += 1
            score = len(_toks(line) & q)
            if score >= need:
                hits.append((-score, order, line))
    if not hits:
        return None
    hits.sort()
    out: list[str] = []
    for _, _, line in hits[:2]:
        if line not in out:
            out.append(line)
    return _clip_spoken(" ".join(out))

_MODE_TOPICS = {"amenities": {"amenities"}, "restaurants": {"dining"}, "nearby": {"nearby"}}
_NOT_FOUND = {
    "amenities": "I couldn't find that among the amenities I have on record. ",
    "restaurants": "I don't have that detail in the restaurant records. ",
    "nearby": "I don't have that in the hotel's records. ",
    "ask": "I'm sorry, I don't have that information. ",
}


async def _handle_hotel_menu(session: Session, user_message: str, clean: str) -> str | None:
    m = session.hotel_menu
    hotel = session.checklist.hotel
    low = " ".join(clean.lower().split())
    bare = low.strip(" .!?")
    uncertain = user_message.lstrip().startswith("[transcription uncertain")
    mode = m["mode"]

    if _CHANGE_HOTEL_RE.search(low):                      # wants a different hotel
        _reset_fields(session, ["hotel"])
        m.update(mode=None, queue=[])
        return await _next_question(session)

    if not uncertain:                                     # "other hotels in Ahmedabad"
        cat = await _catalogue_answer(session, clean)
        if cat:
            return cat + " " + _MENU_NEXT

    choice = _CHOICE_WORDS.get(c.group(1)) if mode is None and (c := _CHOICE_RE.match(bare)) else None
    wants_book = bool(_BOOK_RE.search(low)) and (
        not _NOT_ROOM_RE.search(low) or bool(re.search(r"\b(hotel|room|stay)\b", low)))
    if wants_book or _BOOK_ONLY_RE.match(bare) or choice == "book":
        m.update(done=True, mode=None, queue=[])          # 1. Book this hotel
        return "Great, let's get your booking started. " + await _next_question(session)

    if _MENU_RE.search(low):
        m.update(mode=None, queue=[])
        return _menu_prompt(session, again=True)

    if m["queue"]:                                        # part of an answer is still unread
        ans = _yes_no(bare)
        if _MORE_RE.match(bare) or ans == "yes":
            return _say_more(session)
        if ans == "no":
            m["queue"] = []
            return "No problem. " + _MENU_NEXT

    if mode and (_yes_no(bare) == "no" or _NO_MORE_RE.search(low)):
        m.update(mode=None, queue=[])
        return _menu_prompt(session, again=True)

    # which section is this about?
    section = choice
    if section is None and mode != "ask":
        hits = sorted((rx.search(low).start(), s) for s, rx in _SECTION_RE.items()
                      if s != "ask" and rx.search(low))
        if mode in _SECTION_RE and mode != "ask" and _SECTION_RE[mode].search(low):
            section = mode
        elif hits:
            section = hits[0][1]
        elif _SECTION_RE["ask"].search(low):
            section = "ask"

    if section == "ask" and mode != "ask":                # 5. Ask a question
        m.update(mode="ask", queue=[])
        ans = None if uncertain else _kb_lookup(session, low)
        if ans:
            return _speak_units(session, _sentences(ans), "text")
        return f"What would you like to know about {hotel}?"

    if section and section != mode:                       # open a section
        m.update(mode=section, queue=[])
        return {"amenities": _amenities_intro, "restaurants": _restaurants_intro,
                "nearby": _nearby_intro}[section](session)

    if mode in ("amenities", "restaurants", "nearby"):    # follow-up inside a section
        if _only_topic_words(clean) and not (mode == "nearby" and _DIST_Q.search(low)):
            return {"amenities": _amenities_intro, "restaurants": _restaurants_intro,
                    "nearby": _nearby_intro}[mode](session)
        ans = {"amenities": _amenity_followup, "restaurants": _restaurants_followup,
               "nearby": _nearby_followup}[mode](session, clean)
        if ans:
            return ans

    # a free question (option 5, or anything the section couldn't answer). Inside a
    # section only look further if the question is about some OTHER topic.
    other = okf_store.topics_in(clean) - {"overview"} - _MODE_TOPICS.get(mode, set())
    ans = None if (uncertain or (mode in _MODE_TOPICS and not other)) else _kb_lookup(session, low)
    if ans:
        if mode is None:
            m["mode"] = "ask"
        return _speak_units(session, _sentences(ans), "text")
    if not uncertain and mode in (None, "ask"):           # e.g. "do you have parking?"
        ans = _amenity_followup(session, clean)           # -> only if it is on the hotel's list
        if ans:
            m["mode"] = m["mode"] or "ask"
            return ans
    if mode is None:
        return _menu_prompt(session, lead="Sorry, I didn't catch that. ")
    return _NOT_FOUND[mode] + _MENU_NEXT

    if not uncertain:                                     # nothing matched by topic: search the stored text
        ans = _search_hotel_text(hotel, clean)
        if ans:
            return _speak_units(session, _sentences(ans), "text")
        
def _guard_followup(session: Session) -> str:
    """The real next question, without changing state (sync)."""
    if session.stage == Stage.INFO:
        return _INFO_NEXT
    if session.stage == Stage.POST_BOOKING:
        return _POST_NEXT
    if session.interrupt:
        return _INTERRUPT_PROMPT
    if _menu_pending(session):
        return _menu_prompt(session)
    step = _current_step(session)
    return ("Shall I confirm this booking - yes or no?" if step == "_done"
            else _step_question(session, step))


def _finish(session: Session, text: str) -> str:
    try:                                            # guardrail: never let this break a call
        text, replaced = vg.check_output(
            text, booking_saved=session.stage in (Stage.POST_BOOKING, Stage.COMPLETED))
        if replaced:
            print("  [guardrail] unsafe reply replaced")
            text = f"{text} {_guard_followup(session)}".strip()
    except Exception as exc:
        print(f"  [guardrail] output check skipped: {exc}")
    session.add_message("assistant", text)
    return text

_POST_NEXT = "Is there anything else I can help you with?"
_POST_OPTIONS = (_POST_NEXT + " You can book another hotel, or hear about this "
                 "hotel's amenities, its restaurants or nearby places.")
_ANOTHER_RE = re.compile(
    r"\b(another|new|one more|different|other|second|again|change)\b.*"
    r"\b(hotel|booking|reservation|stay|property|room)s?\b"
    r"|\b(book|reserve)\b.*\b(another|again|new)\b"
    r"|\bbook (a |the )?(hotel|room)\b")
_BYE_RE = re.compile(
    r"\b(bye|goodbye|good bye|that'?s (all|it)|i'?m (done|good)|nothing( else)?|"
    r"no thanks?|no thank you)\b")

async def _handle_post_booking(session: Session, user_message: str, clean: str) -> str:
    m = session.hotel_menu
    low = " ".join(clean.lower().split())
    bare = low.strip(" .!?")
    ans = _yes_no(bare)
    c = _CHOICE_RE.match(bare)
    choice = (_CHOICE_WORDS.get(c.group(1))
              if c and not m["mode"] and not m["queue"] else None)
    wants_book = bool(_BOOK_RE.search(low)) and (
        not _NOT_ROOM_RE.search(low) or bool(re.search(r"\b(hotel|room|stay)\b", low)))

    # 1. Book another hotel: fresh state, back to the city step.
    if choice == "book" or wants_book or _ANOTHER_RE.search(low) or _BOOK_ONLY_RE.match(bare):
        session.start_new_booking()
        return "Sure, let's book another hotel. " + await _next_question(session)

    # 2. Nothing more: end the call (main.py stops on COMPLETED).
    if _BYE_RE.search(low) or (ans == "no" and not m["queue"]):
        session.set_stage(Stage.COMPLETED)
        return "Thank you for calling. Goodbye!"

    # 3. A bare "yes": show the options again.
    if ans == "yes" and not m["queue"]:
        m.update(mode=None, queue=[])
        return "Sure. " + _POST_OPTIONS

    # 4. Amenities / restaurants / nearby places and their follow-ups.
    try:
        reply = await _handle_hotel_menu(session, user_message, clean)
    except Exception as exc:                     # never let this break the call
        print(f"  [post-booking] menu failed: {exc}")
        reply = None
    if not reply:
        return "Sorry, I didn't catch that. " + _POST_OPTIONS
    return reply.replace(_MENU_NEXT, _POST_NEXT)

# ----------------------------------------------------------------------
# Hotel questions in the middle of a booking.
#   question -> "cancel and answer, or continue?"
#   continue -> carry on exactly where we were (session untouched)
#   answer   -> drop the unfinished booking (nothing was saved), stage INFO:
#               questions answered from the hotel's data until "book a room"
# ----------------------------------------------------------------------
_INTERRUPT_TOPICS = {"rating", "overview", "amenities", "dining", "nearby", "wellness",
                     "events", "directions", "pool", "gym", "parking", "wifi",
                     "breakfast", "airport"}
_RATING_RE = re.compile(r"\b(ratings?|stars?)\b")
_INTERRUPT_PROMPT = ("Would you like to cancel the booking and get an answer to your "
                     "question, or continue with the booking?")
_INFO_NEXT = "Anything else I can help you with, or would you like to book a room?"
_INFO_OPTIONS = ("What would you like to know? I can tell you about its amenities, "
                 "restaurants, nearby places or rating, or you can say book a room.")
_CONTINUE_RE = re.compile(r"\b(continue|carry on|proceed|resume|keep going|go on|go ahead|booking)\b")
_ANSWER_RE = re.compile(r"\b(answer|cancel|question|first)\b")
_BOOK_INTENT_RE = re.compile(
    r"\b(book|reserve) (me )?(a |the |this |one )?(room|hotel|stay)\b|\bmake a (booking|reservation)\b")
_INFO_BYE_RE = re.compile(
    r"\b(bye|goodbye|good bye|that'?s (all|it)|nothing( else)?|no thanks?|no thank you)\b")


def _hotel_mentioned(text: str) -> str | None:
    """Canonical name of the ACTIVE hotel the caller named, tolerating speech-to-text
    mishearings ('Sunwear Retreat' -> 'Sunvale Retreat, Ahmedabad')."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    best, best_r = None, 0.75                      # minimum similarity
    for name in (okf_store.store._active or {}).values():
        low = name.lower()
        forms = {" ".join(re.findall(r"[a-z0-9]+", low)),
                 " ".join(re.findall(r"[a-z0-9]+", low.split(",")[0]))}   # with / without city
        for form in forms:
            n = len(form.split())
            for size in (n - 1, n, n + 1):
                if size < 1:
                    continue
                for i in range(len(words) - size + 1):
                    r = difflib.SequenceMatcher(None, " ".join(words[i:i + size]), form).ratio()
                    if r > best_r:
                        best, best_r = name, r
    return best


def _hotel_overview(hotel: str):
    return next((d for d in okf_store.hotel_docs(hotel) if d.type == "hotel_overview"), None)


def _stars_of(doc) -> str | None:
    return next((v for k, v in _doc_facts(doc) if k == "Star rating"), None) if doc else None


def _rating_answer(question: str, hotel: str | None) -> str | None:
    """Star rating from the hotel's overview file (hotels.starRating); with no hotel,
    lists the hotels that have the star count the caller said ('five star')."""
    q = question.lower()
    if hotel:
        stars = _stars_of(_hotel_overview(hotel))
        if not stars:
            return f"I don't have a star rating on record for {hotel}."
        extra = " I don't have guest review scores." if "rating" in q else ""
        return f"{hotel} is a {stars}-star hotel.{extra}"
    want = next((n for n in sorted(_numbers_in(question)) if 1 <= n <= 7), None)
    if want is None:
        return None
    found = []
    for name in (okf_store.store._active or {}).values():
        doc = _hotel_overview(name)
        if _stars_of(doc) == str(want):
            city = next((v for k, v in _doc_facts(doc) if k == "City"), "")
            found.append(name if not city or city.lower() in name.lower() else f"{name} in {city}")
    if not found:
        return f"I don't have any {want}-star hotels on record."
    more = f" and {len(found) - 8} more" if len(found) > 8 else ""
    return f"{want}-star hotels: {_spoken_list(found[:8])}{more}."


def _hotel_question(session: Session, user_message: str, clean: str, step: str) -> dict | None:
    """A hotel question (rating / amenities / restaurants / nearby ...) asked mid-booking."""
    if session.interrupt or not okf_store.ready():
        return None
    if user_message.lstrip().startswith("[transcription uncertain"):
        return None
    rating = bool(_RATING_RE.search(clean.lower()))
    if not (rating or _is_info_question(clean)):
        return None
    topics = okf_store.topics_in(clean) & _INTERRUPT_TOPICS
    if rating:
        topics.add("rating")
    if not topics:
        return None
    if step in _KB_COUNT_STEPS and _numbers_in(clean) and not rating:
        return None                                # "2 rooms please" is an answer
    return {"question": clean, "step": step,
            "hotel": _hotel_mentioned(clean) or session.checklist.hotel}


async def _enter_info_mode(session: Session, q: dict) -> str:
    """Cancels the unfinished booking (nothing was saved) and answers the question."""
    session.interrupt = None
    session.start_new_booking()
    session.set_stage(Stage.INFO)
    session.info_hotel = q["hotel"]
    return await _answer_info(session, q["question"], q["question"])


async def _handle_interrupt(session: Session, user_message: str, clean: str) -> str:
    q, low = session.interrupt, clean.lower()
    if (re.search(r"\b(don'?t|do not|not)\s+cancel\b", low)
            or (_CONTINUE_RE.search(low) and not _ANSWER_RE.search(low))):
        session.interrupt = None
        return "Sure, let's continue. " + await _reask_current(session, _current_step(session))
    if _ANSWER_RE.search(low):
        return await _enter_info_mode(session, q)
    return "Sorry, I didn't catch that. " + _INTERRUPT_PROMPT


async def _answer_info(session: Session, user_message: str, clean: str) -> str:
    m, low = session.hotel_menu, clean.lower()
    named = _hotel_mentioned(clean)
    if named and named != session.info_hotel:
        m.update(mode=None, queue=[])
    if named:
        session.info_hotel = named
    hotel = session.info_hotel
    has_topic = bool(okf_store.topics_in(clean) - {"overview"}) or bool(_RATING_RE.search(low))
    if named and not has_topic and m.get("pending_q"):      # they only named the hotel
        clean = user_message = m.pop("pending_q")
        low = clean.lower()
    cat = await _catalogue_answer(session, clean)
    if cat:
        return cat + " " + _INFO_NEXT
    if _RATING_RE.search(low):
        ans = _rating_answer(clean, hotel)
        if ans:
            return ans + " " + _INFO_NEXT
    if not hotel:
        m["pending_q"] = clean
        return "Which hotel would you like to know about?"
    m.pop("pending_q", None)

    topics = okf_store.topics_in(clean)
    if (topics == {"overview"}) or (named and not has_topic and not m["mode"] and not m["queue"]):
        doc = _hotel_overview(hotel)
        text = doc.spoken.get("en") if doc else ""
        return (text or f"I don't have details on record for {hotel}.") + " " + _INFO_NEXT

    session.checklist.hotel = hotel                        # lets the menu handlers read this hotel
    try:
        reply = await _handle_hotel_menu(session, user_message, clean)
    except Exception as exc:
        print(f"  [info] failed: {exc}")
        reply = None
    finally:
        session.checklist.hotel = None
    return (reply or _INFO_OPTIONS).replace(_MENU_NEXT, _INFO_NEXT)


async def _handle_info_mode(session: Session, user_message: str, clean: str) -> str:
    low = " ".join(clean.lower().split())
    bare = low.strip(" .!?")
    c = _CHOICE_RE.match(bare)
    wants_book = (
        (bool(_BOOK_RE.search(low))
         and (not _NOT_ROOM_RE.search(low) or bool(re.search(r"\b(hotel|room|stay)\b", low))))
        or bool(_BOOK_INTENT_RE.search(low)) or bool(_BOOK_ONLY_RE.match(bare))
        or bool(c and _CHOICE_WORDS.get(c.group(1)) == "book"))
    if wants_book:                                  # fresh booking, back to the city step
        session.start_new_booking()
        session.info_hotel = None
        return "Sure, let's book a room. " + await _next_question(session)
    if _INFO_BYE_RE.search(low) or (_yes_no(bare) == "no" and not session.hotel_menu["queue"]):
        session.set_stage(Stage.ABANDONED)          # main.py ends the call on this stage
        return "Thank you for calling. Goodbye!"
    return await _answer_info(session, user_message, clean)

async def _city_hotels_question(session: Session, clean: str, step: str) -> str | None:
    """'What are the hotels in Ahmedabad?' at the city step: take that city and list its hotels."""
    low = clean.lower()
    if step not in ("city", "hotel") or not re.search(r"\bhotels?\b", low):
        return None
    if re.search(r"\b(ratings?|stars?)\b", low):
        return None                                   # a rating question, handled elsewhere
    if re.search(r"\b(book|want|choose|select|take|prefer|go with)\b", low):
        return None                                   # "I want Vista Haven hotel in Goa" picks a hotel
    if not re.search(r"\b(what|which|list|tell|show|any|other|are there|have)\b", low):
        return None
    await _prefetch(session, "city")
    city = next((c for c in session.available_cities or [] if c.lower() in low), None)
    if not city:
        return None                                   # no city named: normal flow
    out = await EXECUTORS["list_hotels"](city=city)   # pure read
    if "error" in out:
        return None
    await _apply_tool_result(session, "list_hotels", {"city": city}, out)   # sets the city + hotel list
    return await _next_question(session)              # "In Ahmedabad we have ... Which hotel would you like?"

_STRIKES_KEY = "guardrail:injection"


async def generate_reply(session: Session, user_message: str) -> str:
    """Entry point for the conversation loop (input guardrail, then the flow)."""
    try:
        injected = vg.detect_prompt_injection(_strip_markers(user_message))
    except Exception as exc:
        print(f"  [guardrail] input check skipped: {exc}")
        injected = False
    if injected:
        strikes = session.field_attempts.get(_STRIKES_KEY, 0) + 1
        session.field_attempts[_STRIKES_KEY] = strikes
        print(f"  [guardrail] prompt-injection attempt {strikes}/{vg.MAX_INJECTION_STRIKES}")
        session.add_message("user", "[caller message blocked by guardrails]")
        if strikes >= vg.MAX_INJECTION_STRIKES:
            session.set_stage(Stage.TRANSFERRED)      # main.py ends the call on this stage
            return _finish(session, vg.TRANSFER_LINE)
        return _finish(session, vg.INJECTION_DEFLECTION + " " + _guard_followup(session))
    return await _generate_reply_inner(session, user_message)


async def _generate_reply_inner(session: Session, user_message: str) -> str:
    """The original generate_reply, unchanged."""
    session.add_message("user", user_message)
    clean = _strip_markers(user_message)
    if session.stage == Stage.INFO:
        return _finish(session, await _handle_info_mode(session, user_message, clean))
    if session.interrupt:
        return _finish(session, await _handle_interrupt(session, user_message, clean))
    if session.stage == Stage.POST_BOOKING:
        return _finish(session, await _handle_post_booking(session, user_message, clean))
    step = _current_step(session)

    # 0a. Hotel chosen, booking not started yet: the options menu
    #     (amenities / restaurants / nearby / questions / book).
    if _menu_pending(session):
        try:
            menu_reply = await _handle_hotel_menu(session, user_message, clean)
        except Exception as exc:               # never let the menu break a call
            print(f"  [menu] failed, going straight to booking: {exc}")
            session.hotel_menu["done"] = True
            menu_reply = await _next_question(session)
        if menu_reply is not None:
            return _finish(session, menu_reply)
    hq = _hotel_question(session, user_message, clean, step)
    if hq:
        if not any(v is not None for v in session.checklist.to_dict().values()):
            return _finish(session, await _enter_info_mode(session, hq))   # nothing to cancel yet
        session.interrupt = hq                         # session is left untouched until they choose
        return _finish(session, _INTERRUPT_PROMPT)

    ch = await _city_hotels_question(session, clean, step)
    if ch:
        return _finish(session, ch)
    
    # 0. A question the hotel knowledge base can answer
    #    repeat the pending question. Any failure falls through to the
    #    normal booking flow - the KB can never break a call.
    try:
        kb_reply = await _kb_answer(session, user_message, clean, step)
    except Exception as exc:
        print(f"  [kb] lookup failed, continuing normally: {exc}")
        kb_reply = None
    if kb_reply is not None:
        return _finish(session, kb_reply)

    # 1. Everything collected: only the final read-back/yes-no is allowed.
    if step == "_done":
        return _finish(session, await _handle_final(session, clean))

        # 1b. Phone number (asked first thing on the call) - handled in code.
    if step in ("phone", "phone_confirmed"):
        return _finish(session, await _handle_phone(session, clean))

    # 2. A name/email/date is waiting for the caller's yes/no.
    pend = session.pending
    if pend and pend["field"] == step:
        ans = _yes_no(clean)
        if ans == "yes":
            session.pending = None
            session.update_checklist(**{step: pend["value"]})
            session.locked_fields.add(step)
            return _finish(session, await _next_question(session))
        if ans == "no":
            session.pending = None
            return _finish(session, _reask_after_no(session, step))
        session.pending = None      # a correction / new value: read it fresh below
    elif pend:
        session.pending = None

    # 3a. Guest counts are read in code - no model involved.
    if step in ("adults", "children", "rooms"):
        return _finish(session, await _handle_counts(session, step, clean))

    # 3b. A room type named exactly is taken in code (one per room).
    if step == "room_type" and await _take_room_types(session, clean):
        return _finish(session, await _next_question(session))

    # 3. Normal turn: let the model understand the speech, enforce in code.
    turn = _Turn(user_text=clean, rooms_before=len(session.checklist.room_type or []))
    history = session.conversation[-_MAX_HISTORY_MESSAGES:]
    messages = [{"role": "system", "content": _system_prompt(session)}, *history]
    client = _get_client()

    for round_num in range(_MAX_TOOL_ROUNDS):
        step_before = _current_step(session)
        t0 = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=350,
                reasoning_effort="low",
                tools=_tools_for_next_step(session),
                tool_choice="auto",
            )
        except groq.APIError as exc:
            print(f"[ai_response] Groq API error (round {round_num}): {exc}")
            if "tool_use_failed" in str(exc):
                if step_before in _CONFIRM_FIELDS:
                    return _finish(session, await _fallback_for_step(session, step_before, clean, turn))
                return _finish(session, await _next_question(session))
            return _FALLBACK_REPLY
        print(f"  [timing] round {round_num} groq call: {time.perf_counter() - t0:.2f}s")

        choice = response.choices[0].message
        if not choice.tool_calls:
            text = _truncate_after_first_question(_strip_repeated_sentences(choice.content or ""))
            if step_before in _CONFIRM_FIELDS:
                return _finish(session, await _fallback_for_step(session, step_before, clean, turn))
            if (not text or not _reply_fits_step(session, text, step_before)
                    or _skips_ahead(text, step_before)):
                print(f"  [guard] off-step reply replaced; real next step is {step_before}")
                text = await _next_question(session)
            return _finish(session, text)

        tool_messages = await _run_tool_calls(session, choice.tool_calls, turn)
        reply = await _reply_from_turn(session, turn, step_before)
        if reply:
            return _finish(session, reply)

        assistant_msg = choice.model_dump(exclude_none=True)
        assistant_msg["content"] = ""
        assistant_msg.pop("reasoning", None)
        messages.append(assistant_msg)
        messages.extend(tool_messages)

    print("[ai_response] hit max tool-call rounds without a final reply")
    return _finish(session, await _next_question(session))


async def close() -> None:
    """Closes the underlying HTTP connections while the event loop is
    still running (avoids the noisy 'Event loop is closed' on Windows)."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None