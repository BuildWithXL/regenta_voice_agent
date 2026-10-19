"""
Voice-agent guardrails - the chatbot's guardrails.py rules, adapted for the
phone flow. Pure functions only: no Mongo, no session, no I/O, so importing
this can never change how the booking flow behaves.

Mapping from the chatbot (guardrails.py / hotel_agent.py):

  chatbot                              voice
  -----------------------------------  ----------------------------------------
  detect/sanitize_injection            detect_prompt_injection (same patterns).
                                       The voice agent BLOCKS the turn instead
                                       of wrapping it in "[USER SAID]:".
  filter_leaked_prompt                 check_output (leak markers of the voice
                                       prompt)
  validate_output                      check_output (HTML strip, false
                                       "booking confirmed" claims, length cap)
  verify_name / verify_name_part       verify_name / verify_name_part
  verify_email                         verify_email
  verify_phone                         verify_phone (10-digit Indian mobile,
                                       libphonenumber when installed)
  verify_room_selection                check_booking_ready (hotel / room type
                                       must be ones the DB listed this call)
  verify_room_occupancy                check_booking_ready (3 adults / 2
                                       children per room ceiling)

Not ported on purpose (no equivalent in the voice flow): room-selection
locking and the duplicate-confirm race guard (nothing client-supplied, one
caller, turns are sequential), booking-reference rate limiting, cancellation
rules, and email OTP verification (that would be a new voice step).
"""
from __future__ import annotations

import os
import re
import unicodedata
from datetime import date, datetime

try:                                    # same library the chatbot uses
    import phonenumbers
except ImportError:                     # still works without it (see verify_phone)
    phonenumbers = None

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
MIN_NAME_PART_LEN = int(os.getenv("VOICE_MIN_NAME_LEN", "3"))   # chatbot rule: 3
MAX_REPLY_CHARS = 1200                  # chatbot caps at 2000; speech needs far less
MAX_INJECTION_STRIKES = 3               # attempts before the call is handed over

INJECTION_DEFLECTION = "I can only help with hotel bookings and hotel information."
TRANSFER_LINE = ("I'm sorry, I can't continue with this request. "
                 "Let me transfer you to a team member. Goodbye.")
_SAFE_REPLACEMENT = "Sorry, let me stay on your booking."

# ----------------------------------------------------------------------
# 1. Input: prompt injection (patterns copied from the chatbot)
# ----------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (your|the|all) (instructions|rules|prompt)",
    r"forget (your|all|the) (instructions|rules|training|system prompt)",
    r"\byou are now\b",
    r"\bact as (if )?(a|an)\b",
    r"pretend (you are|to be)",
    r"new instructions",
    r"override (your|the) (instructions|rules|system)",
    r"reveal (your|the) (system prompt|instructions|rules)",
    r"what (is|are) your (system prompt|instructions)",
    r"repeat (your|the) (system prompt|instructions) (word for word|verbatim)",
    r"system prompt",
    r"jailbreak",
    r"developer mode",
    r"do anything now",
]
_injection_re = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(text: str) -> bool:
    """True if the caller's words contain a known jailbreak/injection pattern."""
    return bool(_injection_re.search(text or ""))


# ----------------------------------------------------------------------
# 2. Field validators (same rules as the chatbot)
# ----------------------------------------------------------------------
def _name_word_ok(word: str) -> bool:
    if len(word) < MIN_NAME_PART_LEN or not word[0].isalpha():
        return False
    return all(ch.isalpha() or ch in "'-" or unicodedata.category(ch).startswith("M")
               for ch in word)


def verify_name_part(value: str) -> bool:
    """One name field (first OR last): every word has at least
    MIN_NAME_PART_LEN letters; no digits or symbols."""
    words = (value or "").split()
    return bool(words) and all(_name_word_ok(w) for w in words)


def verify_name(name: str) -> bool:
    """Full name: at least a first and a last part, each valid."""
    parts = (name or "").split()
    return len(parts) >= 2 and all(_name_word_ok(p) for p in parts)


def verify_email(email: str) -> bool:
    """Shape check only (same regex as the chatbot)."""
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$", email or ""))


def _ten_digits(raw) -> str | None:
    digits = re.sub(r"\D", "", str(raw or ""))
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits if len(digits) == 10 else None


def verify_phone(phone) -> bool:
    """A real 10-digit Indian mobile (the voice flow stores 10 digits).
    Uses libphonenumber when installed; always rejects the chatbot's fake
    patterns: all-same digits (0000000000) and sequential runs
    (1234567890, 9876543210)."""
    digits = _ten_digits(phone)
    if digits is None:
        return False
    if phonenumbers is not None:
        try:
            if not phonenumbers.is_valid_number(phonenumbers.parse(digits, "IN")):
                return False
        except phonenumbers.NumberParseException:
            return False
    elif digits[0] not in "6789":       # fallback: Indian mobiles start 6-9
        return False
    if len(set(digits)) == 1:
        return False
    nums = [int(d) for d in digits]
    ascending = all((b - a) % 10 == 1 for a, b in zip(nums, nums[1:]))
    descending = all((a - b) % 10 == 1 for a, b in zip(nums, nums[1:]))
    return not (ascending or descending)


# ----------------------------------------------------------------------
# 3. Pre-save booking check (what hotel_agent.py runs before save_booking)
# ----------------------------------------------------------------------
def _names_of(items) -> set[str]:
    return {x.strip().lower() for x in (items or []) if isinstance(x, str)}


def check_booking_ready(c, available_hotels=None, available_room_types=None) -> list[str]:
    """Checks the collected checklist `c` just before the booking is saved.
    Returns the checklist field names that fail (empty list = fine), so the
    caller can reset exactly those and ask again.  Lists the DB gave us this
    call are used when present; if we never had them, that check is skipped
    rather than blocking a good booking."""
    bad: list[str] = []

    if not verify_name_part(getattr(c, "first_name", "") or ""):
        bad.append("first_name")
    if not verify_name_part(getattr(c, "last_name", "") or ""):
        bad.append("last_name")
    if not verify_email(getattr(c, "email", "") or ""):
        bad.append("email")
    if not verify_phone(getattr(c, "phone", "") or ""):
        bad.append("phone")

    try:
        ci = datetime.strptime(c.check_in, "%Y-%m-%d").date()
        co = datetime.strptime(c.check_out, "%Y-%m-%d").date()
        if ci < date.today():
            bad.append("check_in")
        if co <= ci:
            bad.append("check_out")
    except (TypeError, ValueError):
        bad += [f for f in ("check_in", "check_out") if f not in bad]

    adults, children, rooms = (getattr(c, "adults", None), getattr(c, "children", None),
                               getattr(c, "rooms", None))
    ok_counts = (isinstance(adults, int) and isinstance(children, int)
                 and isinstance(rooms, int) and rooms >= 1 and adults >= 1
                 and children >= 0 and adults <= 3 * rooms and children <= 2 * rooms)
    if not ok_counts:
        bad += ["adults", "children", "rooms"]

    hotels = _names_of(available_hotels)
    if hotels and str(getattr(c, "hotel", "") or "").strip().lower() not in hotels:
        bad.append("hotel")

    types = _names_of(available_room_types)
    chosen = getattr(c, "room_type", None)
    if not isinstance(chosen, list) or not chosen:
        if "room_type" not in bad:
            bad.append("room_type")
    elif types and any(str(t).strip().lower() not in types for t in chosen):
        bad.append("room_type")

    return bad


# ----------------------------------------------------------------------
# 4. Output: what the caller is about to hear
# ----------------------------------------------------------------------
_LEAK_MARKERS = ("system_prompt", "booking phone agent", "next question - ask only",
                 "one short sentence plus one question", "never call a tool with")

# First-person claims that a booking exists. Third-person policy text from the
# knowledge base ("bookings are confirmed by email") deliberately does not match.
_FALSE_CONFIRM_RE = re.compile(
    r"\byour (booking|reservation)\b[^.?!]{0,30}\b(is|has been|was)( now)?\s+"
    r"(confirmed|complete|completed|done)\b"
    r"|\bsuccessfully booked\b"
    r"|\byour booking (id|reference|number) is\b"
    "|\U0001F389",
    re.IGNORECASE,
)


def check_output(reply: str, booking_saved: bool) -> tuple[str, bool]:
    """Final safety pass on a reply before it is spoken.

    Returns (text, replaced). replaced=True means the reply was thrown away
    and swapped for a neutral line - the caller should then append the real
    next question so the call keeps moving."""
    if not reply or not str(reply).strip():
        return _SAFE_REPLACEMENT, True
    text = str(reply)

    low = text.lower()
    if any(m in low for m in _LEAK_MARKERS):           # leaked prompt text
        return _SAFE_REPLACEMENT, True

    text = re.sub(r"<[^>]*>", "", text).strip()         # TTS would read tags aloud

    if not booking_saved and _FALSE_CONFIRM_RE.search(text):
        return _SAFE_REPLACEMENT, True                  # nothing was saved

    if len(text) > MAX_REPLY_CHARS:
        cut = text[:MAX_REPLY_CHARS]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        text = cut[: end + 1] if end > MAX_REPLY_CHARS // 2 else cut.rsplit(" ", 1)[0]

    return (text.strip() or _SAFE_REPLACEMENT), not text.strip()