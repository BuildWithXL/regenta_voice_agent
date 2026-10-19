"""
Phase 7 — session data + storage.

Merged from what used to be two files:
  - models.py  (Session / Stage / Checklist dataclasses)
  - store.py   (SessionStore — in-memory, asyncio.Lock-guarded storage)

They're combined here because they're small and always used together;
nothing else needed to change since every name below (Session, Stage,
Checklist, SessionStore, store) still means exactly what it did before
-- only the import path changed, e.g. `from models import Stage` is now
`from session import Stage`.

Deliberately plain dataclasses for Session/Checklist, no MongoDB/Pydantic
coupling here -- mongo.py is the only place that knows how a Session
gets turned into a stored document.

Storage is a plain dict guarded by an asyncio.Lock so concurrent calls
(multiple customers at once) can't corrupt each other's entries. This
is intentionally process-local: if/when this moves onto real telephony
with multiple worker processes, swap SessionStore for something shared
(Redis, etc.) behind the same interface -- nothing above this module
needs to change.
Session data + in-memory storage for support calls.

Plain dataclasses, no MongoDB coupling: mongo.py is the only place that knows
how a Session is turned into a stored document. Storage is a dict guarded by
an asyncio.Lock; it is process-local, so swap SessionStore for something
shared (Redis, etc.) if you run several worker processes.

session.py imports from models.py; models.py must never import session.py.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from config import settings


class Stage(str, Enum):
    """Where a given conversation currently is. Doc's Phase 3/4/6 flow
    maps onto this loosely — code driving the AI loop is what actually
    moves a session between stages; this enum just records it."""

    COLLECTING_INFO = "collecting_info"       # Phase 3: still filling checklist
    CHECKING_AVAILABILITY = "checking_availability"  # Phase 4/5: tool call in flight
    AWAITING_CONFIRMATION = "awaiting_confirmation"  # Phase 6: quoted, waiting on customer
    BOOKING = "booking"                       # Phase 5: create_booking tool call in flight
    POST_BOOKING = "post_booking"             # booking saved; offering further help
    INFO = "info"                             # answering hotel questions; no booking in progress
    COMPLETED = "completed"                   # Phase 7: booking confirmed + saved
    TRANSFERRED = "transferred_to_human"      # transfer_to_human was used
    ABANDONED = "abandoned"                   # session went idle / call dropped


# REPLACE the whole Checklist class with:
@dataclass
class Checklist:
    """One field per step of the fixed booking flow, in the order they
    must be collected. `None` means not collected yet."""

    city: str | None = None
    hotel: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    check_in: str | None = None
    check_out: str | None = None
    email: str | None = None
    phone: str | None = None
    phone_confirmed: bool | None = None
    adults: int | None = None
    children: int | None = None
    rooms: int | None = None
    room_type: list[str] | None = None   # one entry per room

    def missing_fields(self) -> list[str]:
        return [f for f, v in self.to_dict().items() if v is None]

    def is_complete(self) -> bool:
        return len(self.missing_fields()) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "hotel": self.hotel,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "check_in": self.check_in,
            "check_out": self.check_out,
            "email": self.email,
            "phone": self.phone,
            "phone_confirmed": self.phone_confirmed,
            "adults": self.adults,
            "children": self.children,
            "rooms": self.rooms,
            "room_type": self.room_type,
        }

from models import SupportStage, SupportTicket


@dataclass
class Session:
    """One customer's call/conversation, per the doc's Phase 7 spec:
    a unique ID, the full conversation, the checklist, and the current
    stage — all kept separate from every other customer's session."""
    """One customer's call: ID, conversation, stage and support state."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    conversation: list[dict[str, str]] = field(default_factory=list)
    checklist: Checklist = field(default_factory=Checklist)
    stage: Stage = Stage.COLLECTING_INFO

    # Filled in once create_booking (Phase 5) succeeds; this is what
    # eventually gets written to MongoDB.
    booking: dict[str, Any] | None = None
    # Bookings already confirmed and saved in this call (a new one can follow).
    completed_bookings: list = field(default_factory=list)
    interrupt: dict | None = None      # hotel question waiting for "cancel and answer, or continue?"
    info_hotel: str | None = None      # hotel the caller is asking about in INFO stage
    selected_hotel: dict | None = None  # regenta_flow: {"id", "name", "city"} the caller picked
    transfer_number: str | None = None  # number a transfer dials (None: HUMAN_AGENT_NUMBER)
    regenta: dict = field(default_factory=dict)  # regenta_flow: catalogue, city, hotel list, pending transfer
    call_language: str | None = None   # languages.json key the caller chose ("en"/"hi"/"gu"); None until then
    language_attempts: int = 0         # unclear answers to "which language?"
    language: str = "en"     # language of the caller's most recent utterance
    available_hotels: list = None  # populated from the last list_hotels call
    available_cities: list = None  # populated from the last list_cities call
    available_room_types: list = None  # populated from the last get_room_options call
    # The number this call came in on: Plivo's "From" (plivo_server.py), or
    # --caller-number in local mic/file tests. None if unknown.
    caller_number: str | None = None

    # How many guesses have been made per field (first_name, last_name,
    # email) - tracked in code via record_field_attempt, not left to the
    # model to count from conversation history.
    field_attempts: dict[str, int] = field(default_factory=dict)

    # Fields whose value is now FINAL — either the customer explicitly
    # confirmed it (confirm_field) or the retry limit forced acceptance
    # (record_field_attempt). Once a field is in here, its value lives
    # in checklist and is never re-derived from chat history, and the
    # model is told (via the tool result) to stop asking about it —
    # this is what actually prevents the "Trisha/Krishna/Grisha" drift,
    # since it's enforced in code, not just requested in the prompt.
    locked_fields: set[str] = field(default_factory=set)

    # A name/email/date the customer has been asked to confirm but has
    # not yet said yes to: {"field": ..., "value": ...}. Only a "yes" in
    # a LATER message (checked in code) locks it - the model can't.
    pending: dict[str, str] | None = None
    # True once the full booking has been read back for the final yes/no.
    readback_done: bool = False
    # True while we're waiting to hear which detail to change.
    awaiting_change: bool = False
    # Phone step at the start of the call: None | "asked_best" |
    # "need_number" | "confirm_number" | "done"
    phone_state: str | None = None
    phone_candidate: str | None = None    # number waiting for the caller's yes
    # room type -> nightly price, from the last get_room_options call.
    room_prices: dict = field(default_factory=dict)
    # room type -> (max adults, max children) per room, from the DB.
    room_capacity: dict = field(default_factory=dict)

    # Options menu shown right after the hotel is chosen (see ai_response.py).
    hotel_menu: dict = field(default_factory=lambda: {
        "done": False,     # True once the caller picked "Book this hotel"
        "mode": None,      # None | "amenities" | "restaurants" | "nearby" | "ask"
        "queue": [],       # unread rest of a long answer (read on "more")
        "kind": "text",    # how that queue is spoken: "text" | "items"
    })

    def add_message(self, role: str, content: str) -> None:
        """role: 'user' | 'assistant' | 'tool' — matches what Phase 2
        needs to build the message list sent to Groq."""
        self.conversation.append({"role": role, "content": content})
        self.touch()

    def update_checklist(self, **fields: Any) -> None:
        for key, value in fields.items():
            if not hasattr(self.checklist, key):
                raise ValueError(f"Unknown checklist field: {key}")
            setattr(self.checklist, key, value)
        self.touch()

    stage: Stage = Stage.COLLECTING_INFO

    # Support state
    verified_identity: bool = False          # caller matched booking ID + email
    verified_email: str | None = None   # never sent to the model
    support_stage: SupportStage = SupportStage.NONE
    support_ticket: SupportTicket | None = None
    awaiting_transfer: bool = False          # asked "transfer to a human?", waiting for yes/no
    transfer_tool_called: bool = False       # model called offerHumanTransfer this turn
    tool_context: dict[str, Any] = field(default_factory=dict)  # latest result per tool, re-fed to the model
    clarification_failures: int = 0          # consecutive low-confidence STT turns
    failed_verifications: int = 0            # wrong booking ID / email attempts this call
    last_failed_lookup: tuple[str, str] | None = None
    lk_id: str | None = None        # booking ID the caller said (BK-XXXXXXXX)
    lk_email: str | None = None     # email the caller said
    lk_wait: str | None = None      # "id" | "id_confirm" | "email" | "email_confirm"
    backend_failures: int = 0                # tool or LLM errors this call
    injection_attempts: int = 0              # caller turns flagged as prompt injection
    guardrail_blocks: int = 0                # replies blocked / tool data removed by guardrails
    tool_calls_made: int = 0
    facts: list[str] = field(default_factory=list)            # tool results this call (grounding source)
    hotel_topics_found: set[str] = field(default_factory=set)  # getHotelInfo topics with a real answer
    escalation_reason: str | None = None
    handoff_summary: str | None = None       # context passed to the human agent
    language: str = "en"                     # "en" or "hi"; follows the caller's last turn

    # Which flow handles the call (ai_response.py): None until the first
    # caller turn, then "booking" or "support".
    flow: str | None = None
    wants_new_booking: bool = False          # support model called startNewBooking

    def add_message(self, role: str, content: str) -> None:
        """role: 'user' | 'assistant' | 'tool'."""
        self.conversation.append({"role": role, "content": content})
        self.touch()

    def set_stage(self, stage: Stage) -> None:
        self.stage = stage
        self.touch()

    def set_support_stage(self, support_stage: SupportStage) -> None:
        self.support_stage = support_stage
        self.touch()

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def idle_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.updated_at).total_seconds()

    def start_new_booking(self) -> None:
        """Archives the finished booking and clears everything booking-specific,
        so the next booking starts at the city step. The saved Mongo document is
        never touched. Kept: session_id, conversation, language, caller_number,
        available_cities."""
        if self.booking is not None:
            self.completed_bookings.append(self.booking)
        self.booking = None
        self.checklist = Checklist()
        self.stage = Stage.COLLECTING_INFO
        self.field_attempts = {}
        self.locked_fields = set()
        self.pending = None
        self.readback_done = False
        self.awaiting_change = False
        self.phone_state = None
        self.phone_candidate = None
        self.available_hotels = None
        self.available_room_types = None
        self.room_prices = {}
        self.room_capacity = {}
        self.hotel_menu = {"done": False, "mode": None, "queue": [], "kind": "text"}
        self.touch()

class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> Session:
        session = Session()
        async with self._lock:
            self._sessions[session.session_id] = session
        return session

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def get_or_create(self, session_id: str | None) -> Session:
        """Convenience for the call-handling loop: pass None on a
        brand-new call, pass the existing ID on every turn after."""
        if session_id is not None:
            existing = await self.get(session_id)
            if existing is not None:
                return existing
        return await self.create()

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def all_sessions(self) -> list[Session]:
        async with self._lock:
            return list(self._sessions.values())

    async def cleanup_idle(self) -> list[str]:
        """Drops sessions idle longer than SESSION_IDLE_TIMEOUT_SECONDS.
        Not run automatically — call this on a timer from whatever
        eventually owns the call-handling loop (main.py / Voice Gateway)."""
        """Drop sessions idle longer than SESSION_IDLE_TIMEOUT_SECONDS."""
        removed: list[str] = []
        async with self._lock:
            for sid, session in list(self._sessions.items()):
                if session.idle_seconds() > settings.SESSION_IDLE_TIMEOUT_SECONDS:
                    del self._sessions[sid]
                    removed.append(sid)
        return removed


# Single shared instance for the app, same pattern as tts_client's
# module-level _client.
store = SessionStore()