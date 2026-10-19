"""
Regenta call flow: greeting -> city -> hotel -> questions about that hotel ->
transfer to the hotel's reservations / restaurant / banquets team.

  * Cities and hotels come from MongoDB (active only); nothing is invented.
  * Questions are answered only from the selected hotel's knowledge files
    (built from its Mongo record) through booking_agent's info mode.
  * The bot never takes a booking. A room, table or banquet booking is a
    transfer; the number is read from Mongo at transfer time:
        room booking -> hotels.Mobile
        restaurant   -> restaurants.mobile (for that hotel)
        banquets     -> hotels.Mobile
    plivo_server.py dials session.transfer_number once the stage is TRANSFERRED.

ai_response.py routes calls here; existing bookings still go to support_agent.py.
"""
from __future__ import annotations

import difflib
import re

import booking_agent as ba
import mongo
import okf_store
import regenta_topics as topics
import voice_guardrails as vg
from session import Session, Stage

GREETING = "Thank you for calling Regenta. May I know which city you are looking for?"
_ASK_CITY = "May I know which city you are looking for?"
_ANYTHING_ELSE = "Is there anything else I can help you with?"
_HOTELS_PER_TURN = 3
_NOT_CAUGHT = "Sorry, I didn't catch that. "     # info mode's lead when nothing in the records matched

# team -> how it is spoken (numbers: see _transfer)
_TEAMS = {
    "booking": "reservations team",
    "restaurant": "restaurant team",
    "banquets": "banquets team",
}

# Other names / spellings / Hindi for cities -> the English title used in Mongo.
_CITY_ALIASES = {
    "bombay": "Mumbai", "mumbai": "Mumbai", "baroda": "Vadodara", "amdavad": "Ahmedabad",
    "ahmadabad": "Ahmedabad", "calcutta": "Kolkata", "madras": "Chennai", "bangalore": "Bengaluru",
    "poona": "Pune", "bhooj": "Bhuj", "gurgaon": "Gurugram", "new delhi": "Delhi", "delhi": "Delhi",
    "मुंबई": "Mumbai", "मुम्बई": "Mumbai", "बॉम्बे": "Mumbai", "अहमदाबाद": "Ahmedabad",
    "अमदावाद": "Ahmedabad", "वडोदरा": "Vadodara", "बड़ौदा": "Vadodara", "सूरत": "Surat",
    "भुज": "Bhuj", "भरूच": "Bharuch", "जामनगर": "Jamnagar", "सोमनाथ": "Somnath",
    "वापी": "Vapi", "गांधीनगर": "Gandhinagar", "गोवा": "Goa", "जयपुर": "Jaipur",
    "दुबई": "Dubai", "दिल्ली": "Delhi",
}

_BOOK_RE = re.compile(
    r"\b(book|booking|reserve|reservation|host|organi[sz]e|arrange)\b|बुक|बुकिंग"
    r"|\b(want|need|like|looking for|get)\b.{0,25}\b(table|room|rooms|banquet|hall|venue)\b"
    r"|\b(connect|transfer|speak|talk)\b.{0,30}\b(reservation|booking|restaurant|banquet|team)")
_BANQUET_RE = re.compile(
    r"\b(banquet\w*|wedding|reception|events?|party|parties|conference|function|engagement|hall|venue)\b|शादी|बैंक्वेट")
_RESTAURANT_RE = re.compile(r"\b(restaurant\w*|table|dinner|lunch|dine|dining)\b")
_ASKING_RE = re.compile(      # "I want to know about rooms" is a question, not a booking
    r"\b(know|about|tell|info\w*|details?|what|which|how|types?|prices?|cost|rates?)\b")
_MORE_RE = re.compile(r"\b(more|rest|others|other hotels|next|continue|go on)\b")
_SWITCH_RE = re.compile(r"\b(change|switch|instead|another|different|other|how about|what about|looking for|in)\b")
_ORDINALS = {"first": 0, "1st": 0, "one": 0, "1": 0, "second": 1, "2nd": 1, "two": 1, "2": 1,
             "third": 2, "3rd": 2, "three": 2, "3": 2, "fourth": 3, "4th": 3, "four": 3, "4": 3,
             "fifth": 4, "5th": 4, "five": 4, "5": 4}
_FILLER = {"i", "am", "i'm", "im", "looking", "for", "a", "an", "the", "hotel", "hotels", "in",
           "city", "at", "near", "to", "want", "need", "please", "go", "going", "it's", "its",
           "is", "um", "uh", "yes", "ok", "okay", "my", "me", "we", "are", "would", "like",
           "hello", "hi", "hey", "namaste", "good", "morning", "afternoon", "evening", "thanks",
           "thank", "you", "sorry", "yeah", "haan", "ji", "no", "what", "can", "help", "sure"}
# words shared by many hotel names: on their own they don't pick a hotel
_COMMON = {"hotel", "resort", "resorts", "inn", "place", "central", "suites", "spa", "palace",
           "royal", "orchid", "city", "centre", "center", "grand", "convention", "collection",
           "house", "beach", "road"}
_NAME_SKIP = {"regenta", "the", "and", "by", "of", "at", "in", "de", "z"}   # never match on these
# words of multi-word city titles that are not an area by themselves
_AREA_STOP = {"south", "north", "east", "west", "new", "old", "greater", "city", "nagar", "road",
              "hills", "valley", "the"}


# ------------------------------------------------------------- matching
def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _spoken(name: str) -> str:
    return " ".join(name.split())          # stored names carry odd spaces


def _best(text: str, forms: dict[str, object], min_ratio: float) -> object | None:
    """Option whose name best matches a run of the caller's words (STT-tolerant)."""
    words = _words(text)
    best, best_r = None, min_ratio
    for form, option in forms.items():
        n = len(form.split())
        for size in (n - 1, n, n + 1):
            for i in range(max(0, len(words) - size + 1) if size >= 1 else 0):
                chunk = " ".join(words[i:i + size])
                if len(chunk) < 5 and chunk != form:
                    continue                        # short words only exactly: "go"/"sure" aren't Goa/Surat
                r = 1.0 if chunk == form else difflib.SequenceMatcher(None, chunk, form).ratio()
                if r > best_r:
                    best, best_r = option, r
    return best


async def _catalogue(session: Session) -> dict:
    st = session.regenta
    if "cities" not in st:
        st["cities"], st["hotels"] = await mongo.get_active_cities_and_hotels()
    return st


def _city_named(st: dict, text: str) -> str | None:
    """City (or area, e.g. 'Goa' for South / North Goa) the caller named; an alias
    city we don't have (e.g. Mumbai -> 'Mumbai') is returned too, so we can say so."""
    low = text.lower()
    for alias, title in _CITY_ALIASES.items():
        if (alias in low if not alias.isascii()
                else re.search(rf"\b{re.escape(alias)}\b", low)):
            return next((c for c in st["cities"] if c.lower() == title.lower()), title)
    forms = {" ".join(_words(c)): c for c in st["cities"]}
    for c in st["cities"]:
        for w in _words(c):
            if len(c.split()) > 1 and len(w) >= 3 and w not in _AREA_STOP:
                forms.setdefault(w, w.title())
    return _best(text, forms, 0.8)


def _short_name(h: dict) -> str:
    """'Regenta Inn, Ranip, Ahmedabad' -> 'Regenta Inn, Ranip' (city said separately)."""
    parts = [p.strip() for p in _spoken(h["name"]).split(",")]
    tail, city = parts[-1].lower(), h["city"].lower()
    if len(parts) > 1 and (tail in city or city in tail):
        parts = parts[:-1]
    return ", ".join(parts)


def _word_hit(word: str, said: list[str]) -> bool:
    if len(word) < 5:
        return word in said
    return any(len(s) >= 4 and difflib.SequenceMatcher(None, s, word).ratio() >= 0.8 for s in said)


def _hotel_named(st: dict, text: str, listing: bool) -> dict | None:
    """Hotel the caller named. Most names share words ('Regenta', 'Inn', 'Resort'), so a
    match needs a distinctive word ('Baywatch', 'Ranip') - except while picking from
    the list just read, where 'Regenta Inn' is enough. Ties go to that list."""
    said = _words(text)
    shown = st.get("list") or []
    places = {w for c in st["cities"] for w in _words(c)}     # 'Goa' alone is a city, not a hotel
    best, best_key = None, (0, 0, 0)
    for h in st["hotels"]:
        city = places | set(_words(h["city"]))
        words = [w for w in dict.fromkeys(_words(h["name"])) if w not in _NAME_SKIP]
        rare = sum(_word_hit(w, said) for w in words if w not in _COMMON and w not in city)
        common = sum(_word_hit(w, said) for w in words if w in _COMMON or w in city)
        if not rare and not (listing and h in shown and common):
            continue
        key = (rare, common, h in shown)
        if key > best_key:
            best, best_key = h, key
    return best


def _team_asked(clean: str) -> str | None:
    low = clean.lower()
    if not _BOOK_RE.search(low):
        return None
    if _ASKING_RE.search(low) and not re.search(r"\b(book|booking|reserve|reservation)\b|बुक", low):
        return None
    if _BANQUET_RE.search(low):
        return "banquets"
    if _RESTAURANT_RE.search(low):
        return "restaurant"
    return "booking"


# ------------------------------------------------------------- replies
def _city_hotels(st: dict, city: str) -> list[dict]:
    """Hotels in that city, or in every city of that area ('Goa': South Goa, North Goa,
    and hotels named '..., Goa')."""
    want = city.lower()
    return [h for h in st["hotels"]
            if h["city"].lower() == want or want in _words(h["city"])
            or _spoken(h["name"]).split(",")[-1].strip().lower() == want]


def _read_hotels(st: dict) -> str:
    """Next few hotel names of the chosen city, then 'which one?'."""
    hotels, start = st["list"], st["offset"]
    batch = hotels[start:start + _HOTELS_PER_TURN]
    st["offset"] = start + len(batch)
    names = ba._spoken_list([_short_name(h) for h in batch])
    left = len(hotels) - st["offset"]
    if start == 0 and len(hotels) == 1:
        return f"In {st['city']} we have {names}. Would you like that one?"
    lead = f"In {st['city']} we have {names}" if start == 0 else f"We also have {names}"
    if left > 0:
        return f"{lead}, and {left} more. Which one would you like, or shall I read the rest?"
    return f"{lead}. Which one would you like?"


def _choose_city(st: dict, city: str) -> str:
    hotels = _city_hotels(st, city)
    if not hotels:
        st.pop("city", None)
        return (f"I'm sorry, we don't have a hotel in {city} at the moment. "
                "Which other city would you like?")
    st.update(city=city, list=hotels, offset=0)
    return _read_hotels(st)


async def _select(session: Session, hotel: dict) -> str:
    st = session.regenta
    session.selected_hotel = hotel
    session.info_hotel = hotel["name"]
    session.hotel_menu.update(mode=None, queue=[])
    session.set_stage(Stage.INFO)
    if hotel not in (st.get("list") or []):
        st.update(city=hotel["city"], list=_city_hotels(st, hotel["city"]))
    print(f"  [regenta] selected hotel {hotel['name']} ({hotel['id']})")
    team = st.pop("pending_team", None)
    if team:
        return await _transfer(session, team)
    question = st.pop("pending_q", None)
    info = await _topic_answer(session, hotel, question) if question else None
    if info:
        return f"Sure, {_spoken(hotel['name'])}. {info}"
    return f"Sure, {_spoken(hotel['name'])}. What would you like to know about it?"


async def _topic_answer(session: Session, hotel: dict, question: str) -> str | None:
    """Check-in/out, policies, location, facility timings/numbers (regenta_topics)."""
    st = session.regenta
    cache = st.setdefault("details", {})
    key = str(hotel["id"])
    if key not in cache:
        cache[key] = await mongo.get_hotel_details(hotel["id"])
    res = topics.answer(cache[key], _spoken(hotel["name"]), question)
    if not res:
        return None
    reply, number = res
    if reply.endswith("reservations team?"):
        st["offer"] = "booking"                      # a "yes" connects them
    elif number:
        st["offer"] = number                         # a "yes" / "repeat" reads it again
    elif not reply.endswith("?"):
        reply = f"{reply} {_ANYTHING_ELSE}"
    return reply


def _dial_number(raw) -> str | None:
    digits = re.sub(r"\D", "", str(raw or "")).lstrip("0")    # '0' trunk prefix: national number
    if len(digits) == 10:
        digits = "91" + digits              # stored without country code (Indian numbers)
    return "+" + digits if len(digits) >= 10 else None


async def _transfer(session: Session, team: str) -> str:
    hotel = session.selected_hotel
    name = _spoken(hotel["name"])
    try:
        raw = await (mongo.get_restaurant_mobile(hotel["id"]) if team == "restaurant"
                     else mongo.get_hotel_mobile(hotel["id"]))
    except Exception as exc:
        print(f"  [regenta] transfer number lookup failed: {exc!r}")
        raw = None
    number = _dial_number(raw)
    if not number:
        print(f"  [regenta] WARNING: no {team} number for {hotel['name']} ({hotel['id']}); not transferring")
        return (f"I'm sorry, our {_TEAMS[team]} at {name} can't be reached on this line "
                f"right now. {_ANYTHING_ELSE}")
    session.transfer_number = number
    session.escalation_reason = f"{team}_transfer"
    session.handoff_summary = f"{team} enquiry for {hotel['name']}"
    session.set_stage(Stage.TRANSFERRED)
    print(f"  [regenta] transferring to {team} of {hotel['name']}")
    return f"Sure, I'm connecting you to our {_TEAMS[team]} at {name}. Please hold."


# ------------------------------------------------------------- the turn
def opening_line(session: Session) -> str:
    session.add_message("assistant", GREETING)
    return GREETING


async def generate_reply(session: Session, user_message: str) -> str:
    try:
        injected = vg.detect_prompt_injection(ba._strip_markers(user_message))
    except Exception as exc:
        print(f"  [guardrail] input check skipped: {exc}")
        injected = False
    if injected:
        strikes = session.field_attempts.get(ba._STRIKES_KEY, 0) + 1
        session.field_attempts[ba._STRIKES_KEY] = strikes
        session.add_message("user", "[caller message blocked by guardrails]")
        if strikes >= vg.MAX_INJECTION_STRIKES:
            session.set_stage(Stage.TRANSFERRED)
            return ba._finish(session, vg.TRANSFER_LINE)
        return ba._finish(session, f"{vg.INJECTION_DEFLECTION} {_ASK_CITY if not session.selected_hotel else _ANYTHING_ELSE}")
    session.add_message("user", user_message)
    try:
        reply = await _turn(session, user_message, ba._strip_markers(user_message))
    except Exception as exc:                        # never leave the caller in silence
        print(f"  [regenta] turn failed: {exc!r}")
        reply = "Sorry, I didn't catch that. " + (_ANYTHING_ELSE if session.selected_hotel else _ASK_CITY)
    return ba._finish(session, reply)


async def _turn(session: Session, user_message: str, clean: str) -> str:
    st = await _catalogue(session)
    low = " ".join(clean.lower().split())
    bare = low.strip(" .!?")
    hotel = session.selected_hotel
    yes_no = ba._yes_no(bare)

    # 1. Room / table / banquet booking -> that hotel's team.
    team = _team_asked(clean)
    named_hotel = _hotel_named(st, clean, listing=bool(st.get("city") and not hotel))
    if team and named_hotel:
        session.selected_hotel = hotel = named_hotel
        session.info_hotel = hotel["name"]
    if team and hotel:
        return await _transfer(session, team)
    offered = st.pop("offer", None)
    if team:
        st["pending_team"] = team
        if st.get("city"):
            st["offset"] = 0
            return f"Sure, I can connect you. First, which hotel? {_read_hotels(st)}"
        return f"Sure, I can connect you to the right team. First, {_ASK_CITY[0].lower()}{_ASK_CITY[1:]}"

    if offered == "booking" and hotel and yes_no == "yes":
        return await _transfer(session, "booking")
    if offered and offered != "booking" and (yes_no == "yes" or re.search(r"\b(repeat|again)\b", low)):
        return f"Sure, it's {offered}. {_ANYTHING_ELSE}"
    if offered and yes_no == "no":
        return "No problem. " + _ANYTHING_ELSE

    # 2. Caller is done.
    if ba._INFO_BYE_RE.search(low) or (hotel and yes_no == "no" and not session.hotel_menu["queue"]):
        session.set_stage(Stage.COMPLETED)           # plivo_server hangs up on COMPLETED
        return "Thank you for calling Regenta. Goodbye!"

    asked_first = not hotel and topics.is_topic(clean)
    if asked_first:
        st["pending_q"] = clean                      # answered once a hotel is picked

    # 3. Picking a hotel (by name, by position in the list just read, or "yes" to a single one).
    if named_hotel and named_hotel != hotel:
        return await _select(session, named_hotel)
    listing = st.get("city") and not hotel and st.get("list")
    if listing:
        words = _words(bare)
        pos = next((_ORDINALS[w] for w in words if w in _ORDINALS), None)
        if bare in ("last", "the last one"):
            pos = len(st["list"]) - 1
        if pos is not None and len(words) <= 5 and pos < len(st["list"]):
            return await _select(session, st["list"][pos])
        if yes_no == "yes" and len(st["list"]) == 1:
            return await _select(session, st["list"][0])
        if yes_no == "no" and len(st["list"]) == 1:
            st.pop("city", None)
            return "No problem. Which other city would you like?"
        if (_MORE_RE.search(low) or yes_no == "yes") and st["offset"] < len(st["list"]):
            return _read_hotels(st)

    # 4. Naming a city (or switching city) -> its hotels.
    city = _city_named(st, clean)
    same_area = city == st.get("city") or (hotel and hotel in _city_hotels(st, city or ""))
    if city and not same_area and (not hotel or len(_words(clean)) <= 4 or _SWITCH_RE.search(low)):
        session.selected_hotel = None
        session.info_hotel = None
        session.set_stage(Stage.COLLECTING_INFO)
        return _choose_city(st, city)

    # 5. A question about the selected hotel -> only from its records.
    if hotel and yes_no == "yes":
        return "Sure. " + ba._INFO_OPTIONS
    if hotel:
        info = await _topic_answer(session, hotel, clean)
        if info:
            return info
        session.set_stage(Stage.INFO)
        session.info_hotel = hotel["name"]
        if not okf_store.hotel_docs(hotel["name"]):
            return (f"I'm sorry, I don't have those details for {_spoken(hotel['name'])} "
                    f"on record yet. {_ANYTHING_ELSE}")
        reply = await ba._answer_info(session, user_message, clean)
        reply = reply.replace(_NOT_CAUGHT, "I'm sorry, I don't have that information on record. ")
        if session.info_hotel != hotel["name"]:      # it recognised another hotel
            other = next((h for h in st["hotels"] if h["name"] == session.info_hotel), None)
            if other:
                session.selected_hotel = other
        return reply

    # 6. Still need a city / hotel.
    if asked_first:
        if listing:
            st["offset"] = 0
            return "Sure, I can help with that. First, which hotel? " + _read_hotels(st)
        return f"Sure, I can help with that. First, {_ASK_CITY[0].lower()}{_ASK_CITY[1:]}"
    if listing:
        st["offset"] = 0
        return "Sorry, which hotel would you like? " + _read_hotels(st)
    left = [w for w in _words(clean) if w not in _FILLER]
    if 1 <= len(left) <= 3 and not ba._is_info_question(clean):
        return (f"I'm sorry, we don't have a hotel in {' '.join(left).title()}. "
                "Which other city would you like?")
    return "Sorry, I didn't catch the city. " + _ASK_CITY
