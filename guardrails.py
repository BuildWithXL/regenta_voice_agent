"""
Guardrails for the support agent. No project imports, so it is easy to test.

Three layers, each cheap and deterministic (no extra LLM call):

1. INPUT   check_input()          prompt-injection / off-topic detection on what the
                                  caller said (English, Hindi, romanized Hindi).
2. DATA    sanitize_tool_output() everything read from MongoDB or hotel_info.json is
                                  untrusted: injected text is removed before the model
                                  (or the system prompt) ever sees it.
3. OUTPUT  check_output()         knowledge-based guardrail: the reply may only state
                                  IDs, emails, phone numbers, amounts, hotel times and
                                  policies that came from a tool result or from the
                                  caller; it must not leak internals or promise refunds.

Pattern matching is a safety net, not a guarantee. The real protections are the
code-level guards in ai_response.py (verification, confirmations, internal
arguments); these layers reduce how often the model is pushed toward them.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------
# Text normalisation (defeats zero-width characters, full-width letters, spacing)
# --------------------------------------------------------------------------
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\ufeff\u00ad"))


def _strip_controls(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_INVISIBLE)
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def normalize(text: str) -> str:
    """Single-line, NFKC-normalised, invisible and control characters removed."""
    return re.sub(r"\s+", " ", _strip_controls(text)).strip()


def squash(text: str) -> str:
    """Lowercase letters and digits only: for spacing-insensitive comparisons."""
    return re.sub(r"[^a-z0-9]", "", normalize(text).lower())


def clean_text(text: str, max_chars: int = 1000) -> str:
    """For text we store (ticket descriptions, notes): no control characters, capped."""
    return normalize(text)[:max_chars]


# --------------------------------------------------------------------------
# 1. Input guard
# --------------------------------------------------------------------------
_QUAL = r"(?:all\s+|any\s+|the\s+|your\s+|my\s+|these\s+|those\s+)*(?:(?:previous|prior|above|earlier|system|safety|security|original)\s+)*"
_NOUN = r"(?:instructions?|rules|prompts?|guidelines?|directions|restrictions?|guardrails?|policies)"

_INJECTION = [re.compile(p, re.I) for p in [
    rf"\b(?:ignore|disregard|forget|override|bypass)\s+{_QUAL}{_NOUN}\b",
    r"\b(?:ignore|disregard|forget)\s+(?:everything|all)\s+(?:above|you (?:were|have been) told|you know)\b",
    r"\b(?:reveal|show|tell|repeat|print|read out|display|leak|share)\b.{0,25}\b(?:system|hidden|initial|original|your)\s+(?:prompt|instructions?|configuration|programming)\b",
    r"\bsystem\s+prompt\b", r"\bdeveloper\s+mode\b", r"\bjailbreak", r"\bdan\s+mode\b",
    r"\byou are now\b", r"\bfrom now on,? you\b", r"\bnew instructions?\b",
    r"\bpretend (?:to be|you are|you're)\b", r"\broleplay as\b",
    r"\bact as (?:an? |the )?(?:unrestricted|different|new|admin|developer|root)\b",
    r"\byou (?:must|have to|should) (?:now )?(?:obey|comply)\b",
    r"\b(?:skip|bypass|disable|turn off)\b.{0,15}\b(?:verification|security|checks?|authentication)\b",
    r"\b(?:i am|i'm|this is)\b.{0,15}\b(?:your (?:developer|creator|owner|administrator)|the (?:developer|administrator|sysadmin)|anthropic|openai|groq)\b",
    r"</?\s*(?:system|assistant|instructions?|tool|fetched_data)\b",
    # data-exfiltration probes
    r"\b(?:api[ _-]?keys?|secret keys?|access tokens?|environment variables?|mongodb)\b|\.env\b",
    r"\b(?:list|show|give|dump|read out)\b.{0,15}\b(?:all|every)\b(?!\s+(?:of\s+)?my\b).{0,15}\b(?:bookings?|guests?|customers?|reservations?|emails?)\b",
    r"\b(?:all|every|other)\s+(?:guests?|customers?|bookings?)'?s?\s+(?:details|emails?|names?|bookings?)\b",
    # Hindi (Devanagari)
    r"(?:पिछले|पहले\s*के|सभी|ऊपर\s*के)\s*(?:निर्देश|नियम|आदेश)(?:ों)?.{0,15}(?:भूल|नजरअंदाज|नज़रअंदाज़|इग्नोर|मत\s*मान)",
    r"सिस्टम\s*प्रॉम्प्ट", r"डेवलपर\s*मोड",
    r"(?:अपने|तुम्हारे|आपके)\s*(?:निर्देश|नियम|प्रॉम्प्ट).{0,15}(?:बताओ|बताइए|बताएं|दिखाओ)",
    # romanized Hindi
    r"\b(?:pichle|purane|sabhi)\s+(?:instructions?|rules?|nirdesh)\b.{0,15}\b(?:ignore|bhool|bhul|mat mano)",
    r"\b(?:system prompt|prompt)\s+(?:batao|bataiye|dikhao)\b",
    r"\b(?:instructions?|rules?)\s+(?:ignore|bhool)\s+(?:karo|kar do|jao)\b",
]]
_ROLE_MARKER = re.compile(r"(?:^|\n)\s*(?:system|developer|assistant|tool)\s*:", re.I)
_SQUASHED = ("ignorepreviousinstructions", "ignoreallinstructions", "ignoretheaboveinstructions",
             "disregardpreviousinstructions", "disregardallinstructions", "systemprompt",
             "developermode", "revealyourinstructions", "revealyourprompt", "revealthesystem")

_OFF_TOPIC = re.compile(
    r"\b(?:write|compose|generate|make)\b.{0,15}\b(?:poem|story|essay|song|joke|code|script|program)\b"
    r"|\b(?:python|javascript|sql)\s+(?:code|script|program)\b", re.I)


@dataclass
class InputVerdict:
    kind: str | None   # None (ok) | "injection" | "off_topic"
    cleaned: str       # normalised, length-capped text safe to use downstream


def looks_like_injection(text: str) -> bool:
    raw = _strip_controls(text)
    flat = normalize(text)
    return (bool(_ROLE_MARKER.search(raw)) or any(p.search(flat) for p in _INJECTION)
            or any(s in squash(flat) for s in _SQUASHED))


def check_input(text: str, max_chars: int = 500) -> InputVerdict:
    cleaned = normalize(text)[:max_chars]
    if looks_like_injection(cleaned) or looks_like_injection(text[: max_chars * 2]):
        return InputVerdict("injection", cleaned)
    if _OFF_TOPIC.search(cleaned):
        return InputVerdict("off_topic", cleaned)
    return InputVerdict(None, cleaned)


# --------------------------------------------------------------------------
# 2. Tool-data guard (indirect prompt injection)
# --------------------------------------------------------------------------
def sanitize_tool_output(obj: Any, max_str: int = 1200) -> tuple[Any, int]:
    """Returns (sanitized copy, number of values removed). Strings that look like
    instructions are replaced, angle brackets are neutralised so data can never
    close the <fetched_data> block, and long strings are capped."""
    removed = 0

    def walk(value: Any) -> Any:
        nonlocal removed
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            if looks_like_injection(value):
                removed += 1
                return "[removed: unsafe text]"
            return normalize(value).replace("<", " ").replace(">", " ")[:max_str]
        return value

    return walk(obj), removed


# --------------------------------------------------------------------------
# 3. Output guard (knowledge-based grounding)
# --------------------------------------------------------------------------
_TOOL_NAMES = ("getBookingDetails", "cancelBooking", "createSupportTicket", "getSupportTicket",
               "updateSupportTicket", "getSupportCategories", "getHotelInfo", "transferToHuman",
               "offerHumanTransfer", "endCall")
_LEAK = re.compile(
    "|".join(map(re.escape, _TOOL_NAMES))
    + r"|system\s+prompt|my\s+(?:system\s+)?instructions|api[ _-]?key|\bgsk_[A-Za-z0-9]{8,}|\bsk-[A-Za-z0-9]{8,}"
    + r"|mongodb(?:\+srv)?://|_context|fetched_data|```", re.I)

_BOOKING_ID = re.compile(r"\bB[\s,.-]*K[\s,.-]*(?:[A-Z0-9][\s,.-]*){8}", re.I)
_TICKET_ID = re.compile(r"\bT[\s,.-]*K[\s,.-]*T[\s,.-]*(?:[A-Z0-9][\s,.-]*){8}", re.I)
_EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)
_PHONE = re.compile(r"\+?\d[\d\s-]{8,}\d")
_AMOUNT = re.compile(
    r"(?:₹|rs\.?|inr|\$|usd)\s*(\d[\d,]*(?:\.\d+)?)"
    r"|(\d[\d,]*(?:\.\d+)?)\s*(?:rupees?|रुपये|रुपए|rs\b|inr\b|dollars?|usd\b)", re.I)
_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])")
_TIME_CLAIM = re.compile(
    r"check[\s-]?(in|out)\b[^.?!।]{0,40}?\b(\d{1,2}(?::\d{2})?\s?(?:am|pm|a\.m\.|p\.m\.))", re.I)

_CLOCK_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm|a\.m\.|p\.m\.)", re.I)

# (pattern, KB topic that must have been fetched with getHotelInfo to say this)
_POLICY_CLAIMS = [
    (re.compile(r"\bbreakfast\b[^.?!]{0,25}\b(?:included|free|complimentary|served)\b"
                r"|\b(?:free|complimentary|included) breakfast\b", re.I), "breakfast"),
    (re.compile(r"\bpets?\b[^.?!]{0,20}\b(?:allowed|welcome|permitted)\b", re.I), "pets_policy"),
    (re.compile(r"\bfree cancell?ation\b", re.I), "cancellation_policy"),
    (re.compile(r"\b(?:free|complimentary) parking\b|\bparking\b[^.?!]{0,20}\b(?:available|free)\b", re.I), "parking"),
    (re.compile(r"\b(?:free|complimentary) wi-?fi\b", re.I), "amenities"),
]
_REFUND_DONE = re.compile(r"\brefund\b[^.?!]{0,25}\b(?:has been|have been|is being|was)\s+(?:processed|issued|credited|initiated)\b", re.I)
_REFUND_PROMISE = re.compile(
    r"\b(?:you(?:'ll| will)|we(?:'ll| will)|i(?:'ll| will)|you are going to)\s+(?:definitely |surely )?(?:get|receive|be (?:given|issued))\b[^.?!]{0,20}\brefund\b"
    r"|\bfull refund\b|\bguarantee[sd]?\b[^.?!]{0,20}\brefund\b", re.I)

def _num(s: str) -> str:
    s = s.replace(",", "")
    return s[:-2] if s.endswith(".0") else s


def _tail8(entity: str) -> str:
    s = squash(entity)
    return s[2:] if s.startswith("bk") else s[3:] if s.startswith("tkt") else s

def _clock(s: str) -> str:
    s = re.sub(r"[^a-z0-9:]", "", s.lower())
    return re.sub(r"^0?(\d{1,2})(?::00)?(am|pm)$", r"\1\2", s)

def check_output(reply: str, session: Any, secret_texts: tuple[str, ...] = ()) -> list[str]:
    """Returns the reasons this reply must not be spoken (empty list = fine).
    `session` needs: conversation, facts (tool result JSON strings), hotel_topics_found."""
    problems: list[str] = []
    text = normalize(reply)
    if not text:
        return problems

    # -- leakage of internals
    if _LEAK.search(text):
        problems.append("mentions internal tools, instructions or credentials")
    words = re.findall(r"\w+", text.lower())
    for secret in secret_texts:
        swords = re.findall(r"\w+", secret.lower())
        grams = {tuple(swords[i:i + 8]) for i in range(len(swords) - 7)}
        if any(tuple(words[i:i + 8]) in grams for i in range(len(words) - 7)):
            problems.append("repeats the system instructions")
            break

    # -- grounding corpus: what the caller said + what tools returned this call
    said = " ".join(m["content"] for m in session.conversation if m["role"] == "user")
    facts = " ".join(getattr(session, "facts", []))
    said_variants = [squash(said), squash(re.sub(r"\b(?:at the rate|at|dot)\b", " ", said, flags=re.I))]
    grounded = " ".join(said_variants + [squash(facts)])
    fact_numbers = {_num(n) for n in _NUMBER.findall(facts)}
    fact_times = {_clock(x.group(0)) for x in _CLOCK_TIME.finditer(facts)}
    known_times = fact_times | {_clock(x.group(0)) for x in _CLOCK_TIME.finditer(said)}

    for rx, label in ((_BOOKING_ID, "booking ID"), (_TICKET_ID, "ticket ID")):
        for m in rx.finditer(text):
            if _tail8(m.group(0)) not in grounded:
                problems.append(f"{label} not from the caller or a tool result")
    for m in _EMAIL.finditer(text):
        if squash(m.group(0)) not in grounded:
            problems.append("email not from the caller or a tool result")
    for m in _PHONE.finditer(text):
        if re.sub(r"\D", "", m.group(0)) not in re.sub(r"\D", "", said + " " + facts):
            problems.append("phone number not from the caller or a tool result")
    for m in _AMOUNT.finditer(text):
        if _num(m.group(1) or m.group(2)) not in fact_numbers:
            problems.append("amount not returned by a tool")

    # -- any clock time must come from the caller or a tool result
    for m in _CLOCK_TIME.finditer(text):
        if _clock(m.group(0)) not in known_times:
            problems.append("time not from the caller or a tool result")

    # -- hotel facts must come from getHotelInfo
    topics = getattr(session, "hotel_topics_found", set())
    for m in _TIME_CLAIM.finditer(text):
        topic = "check_in_time" if m.group(1).lower() == "in" else "check_out_time"
        if topic not in topics or _clock(m.group(2)) not in fact_times:
            problems.append(f"{topic.replace('_', ' ')} not from hotel information")
    for rx, topic in _POLICY_CLAIMS:
        if rx.search(text) and topic not in topics:
            problems.append(f"{topic.replace('_', ' ')} claim not from hotel information")

    # -- promises the agent cannot keep
    if _REFUND_DONE.search(text):
        problems.append("claims a refund was processed")
    if _REFUND_PROMISE.search(text) and "cancellation_policy" not in topics:
        problems.append("promises a refund without the cancellation policy")

    return list(dict.fromkeys(problems))
