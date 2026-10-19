"""
Information questions about the selected hotel, answered only from its Mongo
record (mongo.get_hotel_details): check-in/out, policies, location, and
facility timings / contact numbers.

answer() returns None when the question isn't one of these topics; regenta_flow
then falls back to the hotel Q&A it already had. Nothing here is guessed: a
topic with no stored data gets "I don't have that information" plus an offer to
connect to the reservations team (regenta_flow handles the caller's yes).
"""
from __future__ import annotations

import re

OFFER = "I don't have that information. Would you like me to connect you to our reservations team?"

_CHECK_RE = re.compile(r"\bcheck[\s-]?(in|out)s?\b|\bcheckout\b|\bcheckin\b")
_EARLY_LATE_RE = re.compile(r"\b(early|late)\b")
_PET_RE = re.compile(r"\b(pets?|dogs?|cats?|pupp(y|ies))\b")
_POLICY_RE = re.compile(          # policies with no field in the data
    r"\b(ids?|id proof|identity|identification|aadhaa?r|passport|documents?|voter|licen[cs]e)\b"
    r"|\bextra (bed|mattress)|\b(child|children|kids?|infants?|age)\b"
    r"|\bsmok\w*|\bcigarettes?\b|\b(couples?|unmarried|married)\b|\bpolic(y|ies)\b|\brules\b")
_ADDRESS_RE = re.compile(r"\baddress\b|\blocated\b|\blocation\b|\bwhere (is|are)\b|\bwhich area\b|\bpin ?code\b|\bpostal\b")
_DIRECTIONS_RE = re.compile(r"\bdirections?\b|\broute\b|\blandmark\b|\bhow (do|can|to|should) (i |we )?(reach|get|come|go)\b")
_PARKING_RE = re.compile(r"\bpark(ing)?\b|\bvalet\b")
_AIRPORT_RE = re.compile(r"\bairport\b")
_STATION_RE = re.compile(r"\b(railway|train|station|metro|junction)\b")
_TIME_Q_RE = re.compile(r"\b(time|timings?|open\w*|clos\w*|hours?|when)\b")
_NUMBER_Q_RE = re.compile(r"\b(numbers?|contact|phone|call|reach)\b")
_BAR_RE = re.compile(r"\b(bar|pub|lounge)\b")
_RESTAURANT_RE = re.compile(r"\b(restaurants?|dining|breakfast|lunch|dinner|cafe|buffet)\b")
_FACILITIES = (   # facility -> words, hotels flag (0/1+), amenity titles
    ("spa", re.compile(r"\bspa\b|\bmassage\b"), "spa", ("spa",)),
    ("gym", re.compile(r"\bgym\b|\bfitness\b"), "gym", ("fitness center",)),
    ("pool", re.compile(r"\b(swimming )?pool\b|\bswim\w*"), "swimmingPool", ("swimming pool",)),
)


def is_topic(text: str) -> bool:
    """True for a question this module answers (used before a hotel is chosen)."""
    low = text.lower()
    return bool(_CHECK_RE.search(low) or _PET_RE.search(low) or _POLICY_RE.search(low)
                or _ADDRESS_RE.search(low) or _DIRECTIONS_RE.search(low) or _PARKING_RE.search(low)
                or _AIRPORT_RE.search(low) or _STATION_RE.search(low)
                or (_TIME_Q_RE.search(low) or _NUMBER_Q_RE.search(low)) and (
                    _BAR_RE.search(low) or _RESTAURANT_RE.search(low)
                    or any(rx.search(low) for _, rx, _, _ in _FACILITIES)))


# ------------------------------------------------------------- speaking
def spoken_time(hhmm: str) -> str | None:
    """'14:00' -> '2 PM', '11:00' -> '11 in the morning', '23:30' -> '11:30 at night'."""
    m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*$", str(hhmm or ""))
    if not m:
        return None
    h, mins = int(m.group(1)), int(m.group(2) or 0)
    if h > 23 or mins > 59:
        return None
    if (h, mins) == (0, 0):
        return "midnight"
    if (h, mins) == (12, 0):
        return "12 noon"
    h12 = h % 12 or 12
    clock = f"{h12}:{mins:02d}" if mins else str(h12)
    if h < 12:
        return f"{clock} in the morning"
    if h < 17:
        return f"{clock} PM"
    if h < 21:
        return f"{clock} in the evening"
    return f"{clock} at night"


def spoken_number(raw: str) -> str | None:
    """'9825012345' -> '9 8 2 5 0, 1 2 3 4 5' (slow, in two groups)."""
    digits = re.sub(r"\D", "", str(raw or ""))
    if len(digits) < 6:
        return None
    cut = len(digits) - 5 if len(digits) > 10 else len(digits) // 2
    return f"{' '.join(digits[:cut])}, {' '.join(digits[cut:])}"


def _distance(raw: str) -> str | None:
    s = " ".join(str(raw or "").split())
    if not s:
        return None
    s = re.sub(r"\bkms?\b", "km", s, flags=re.I)
    return re.sub(r"\b(mts?|mtrs?)\b", "metres", s, flags=re.I)


_LANDMARK_RE = re.compile(r"^(near|opp\.?|opposite|behind|next to|beside|adjacent|adjoining)\b", re.I)


def _short_address(addr: str, want_pin: bool, full: bool = False) -> str | None:
    """Building, area, first landmark and city ('Opp.' read as 'opposite'); the PIN
    code only when asked, the whole address only when they ask for the full one."""
    s = " ".join(str(addr or "").split()).strip(" ,.")
    if not s:
        return None
    if not want_pin:
        s = re.sub(r"[\s,-]*\b\d{3}\s?\d{3}\b", "", s)
        s = re.sub(r",?\s*India$", "", s, flags=re.I)
    parts = [p.strip(" .-") for p in s.split(",") if p.strip(" .-")]
    if not full and len(parts) > 4:
        mark = next((i for i, p in enumerate(parts) if _LANDMARK_RE.match(p)), None)
        keep = ({0, mark - 1, mark, len(parts) - 1} if mark else {0, len(parts) - 2, len(parts) - 1})
        parts = [parts[i] for i in sorted(keep) if 0 <= i < len(parts)]
    return re.sub(r"\bopp\.?(?=\s)", "opposite", ", ".join(parts), flags=re.I)


# ------------------------------------------------------------- answering
def answer(details: dict, hotel_name: str, text: str) -> tuple[str, str | None] | None:
    """(reply, number read out or None) for an information question, else None."""
    low = text.lower()
    ov, hotel = details["overview"], details["hotel"]
    amen = {a.lower() for a in details["amenities"]}

    # 1. Check-in / check-out
    if _CHECK_RE.search(low):
        if _EARLY_LATE_RE.search(low):
            return OFFER.replace("I don't have that information.",
                                 "I don't have the early check-in or late check-out rules on record."), None
        cin, cout = spoken_time(ov.get("checkin_time")), spoken_time(ov.get("checkout_time"))
        if cin and cout:
            return f"Check-in is at {cin} and check-out is at {cout}.", None
        if cin or cout:
            return (f"Check-in is at {cin}." if cin else f"Check-out is at {cout}."), None
        return OFFER, None

    # 2. Policies (only pets has anything stored: the 'Pet Friendly' amenity)
    if _PET_RE.search(low):
        if "pet friendly" in amen:
            return "Yes, the hotel is listed as pet friendly. I don't have the detailed pet rules.", None
        return OFFER, None

    # 3. Location
    if _AIRPORT_RE.search(low):
        dist, name = _distance(ov.get("airport_distance")), ov.get("airport_title")
        if not dist:
            return OFFER, None
        where = name or "The airport"
        extra = " I don't have the travel time on record." if re.search(r"\b(long|time|minutes?|hours?)\b", low) else ""
        return f"{where} is {dist} from the hotel.{extra}", None
    if _STATION_RE.search(low):
        dist, name = _distance(ov.get("train_distance")), ov.get("train_title")
        if not dist:
            return OFFER, None
        return f"{name or 'The station'} is {dist} from the hotel.", None
    if _PARKING_RE.search(low):
        park, valet = "parking facilities" in amen, "valet service" in amen
        if not (park or valet):
            return OFFER, None
        what = ("parking and valet service" if park and valet
                else "parking" if park else "valet service")
        return f"Yes, the hotel has {what}. I don't have details on whether it is free or paid.", None
    if _DIRECTIONS_RE.search(low):
        addr = _short_address(ov.get("fullAddress"), False)
        how = " ".join(" ".join(str(p).split()) for p in (ov.get("other_title"), ov.get("other_distance")) if p)
        if not (addr or how):
            return OFFER, None
        parts = [f"The hotel is at {addr}." if addr else "", f"{how.strip(' .')}." if how else ""]
        return " ".join(p for p in parts if p), None
    if _ADDRESS_RE.search(low):
        addr = _short_address(ov.get("fullAddress"), bool(re.search(r"pin|postal", low)),
                              full=bool(re.search(r"\b(full|complete|exact)\b", low)))
        return (f"The address is {addr}." if addr else OFFER), None

    # 4. Facilities: timings and contact numbers
    wants_time, wants_number = bool(_TIME_Q_RE.search(low)), bool(_NUMBER_Q_RE.search(low))
    if _BAR_RE.search(low) or _RESTAURANT_RE.search(low):
        if not (wants_time or wants_number):
            return None                              # menus / cuisine: the existing Q&A
        rest = details["restaurants"]
        if _BAR_RE.search(low):
            rest = [r for r in rest if _BAR_RE.search(f"{r.get('name', '')} {r.get('serves', '')}".lower())]
        if not rest:
            return (f"I don't have a {'bar' if _BAR_RE.search(low) else 'restaurant'} on record "
                    f"for {hotel_name}. Is there anything else I can help you with?"), None
        named = [r for r in rest if r.get("name") and r["name"].lower() in low]
        rest = named or rest
        if wants_number:
            r = next((r for r in rest if spoken_number(r.get("mobile"))), None)
            if not r:
                return f"I don't have a contact number for {rest[0].get('name', 'the restaurant')} on record.", None
            return (f"You can reach {r.get('name', 'the restaurant')} on {spoken_number(r['mobile'])}. "
                    "Would you like me to repeat that?"), spoken_number(r["mobile"])
        lines = []
        for r in rest[:2]:
            o, c = spoken_time(r.get("open_time")), spoken_time(r.get("close_time"))
            name = r.get("name") or "The restaurant"
            lines.append(f"{name} is open from {o} to {c}." if o and c
                         else f"I don't have the timings for {name} on record.")
        if len(rest) > 2:
            lines.append(f"There {'is' if len(rest) == 3 else 'are'} {len(rest) - 2} more outlet"
                         f"{'' if len(rest) == 3 else 's'}; just ask me by name.")
        return " ".join(lines), None
    for label, rx, flag, titles in _FACILITIES:
        if rx.search(low):
            has = (hotel.get(flag) or 0) > 0 or any(t in amen for t in titles)
            if not has:
                return (f"I don't have a {label} on record for {hotel_name}. "
                        "Is there anything else I can help you with?"), None
            missing = " or contact number" if wants_number else ""
            return f"The hotel has a {label}, but I don't have its timings{missing} on record.", None
    if wants_number and re.search(r"\b(hotel|reception|front desk)\b", low):
        num = spoken_number(hotel.get("Mobile"))
        if num:
            return f"You can reach {hotel_name} on {num}. Would you like me to repeat that?", num

    # 2b. Any other policy question: nothing stored
    if _POLICY_RE.search(low):
        return OFFER, None
    return None
