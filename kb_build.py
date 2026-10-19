#!/usr/bin/env python3
"""
kb_build.py - generate the hotel knowledge base (Markdown) from MongoDB.

MongoDB is the source of truth and is NEVER written to:
  * the only Mongo calls are find / find_one / count_documents / ping /
    list_collection_names, exposed through a wrapper that has no other
    methods (see ReadOnlyDB);
  * only cities, hotels, rooms, rateplans and the collections named in
    kb_config.py (amenities, services, wellnesses, cuisines, restaurants,
    nearbyhotels) are read - never bookings, customers or credentials (BLOCKED);
  * additionally, point KB_MONGODB_URI at a Mongo user that only has the
    built-in `read` role, so the database itself enforces it.

Usage:
    python kb_build.py --inspect            # show real field names (no files written)
    python kb_build.py --inspect --collections amenities,services   # sample more lookup collections
    python kb_build.py --dry-run            # show what would change
    python kb_build.py                      # KB_STORAGE=local: write ./hotel_kb
                                            # KB_STORAGE=s3:    build, then upload to S3
    python kb_build.py --no-publish         # s3 mode: build locally only

Re-running is safe and cheap: unchanged files are left byte-for-byte alone
(the timestamp only moves when a file's content changes), so git diffs show
exactly what changed in Mongo. Files stay under hotel_kb/hotels/<slug>/.
Generated files carry `generated: true`; only those are ever pruned, so
hand-written files in the same folder are left alone.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kb_config as cfg
import kb_storage

LANGS = ("en",)
_LANG_KEYS = {"en": "en", "english": "en"}
_END = {"en": "."}
_TS = "@@TS@@"


# ---------------------------------------------------------------------------
# Read-only Mongo access
# ---------------------------------------------------------------------------
class ReadOnlyCollection:
    """Exposes ONLY read methods. No insert/update/delete/aggregate/bulk."""
    __slots__ = ("_c",)

    def __init__(self, coll):
        self._c = coll

    def find(self, *args, **kwargs):
        return self._c.find(*args, **kwargs)

    def find_one(self, *args, **kwargs):
        return self._c.find_one(*args, **kwargs)

    def count_documents(self, *args, **kwargs):
        return self._c.count_documents(*args, **kwargs)


class ReadOnlyDB:
    __slots__ = ("_db",)

    def __init__(self, db):
        self._db = db

    def __getitem__(self, name: str) -> ReadOnlyCollection:
        return ReadOnlyCollection(self._db[name])

    def list_collection_names(self) -> list[str]:
        return self._db.list_collection_names()


def connect() -> tuple[Any, ReadOnlyDB]:
    from pymongo import MongoClient          # installed with motor
    from config import settings
    uri = os.getenv("KB_MONGODB_URI") or os.getenv("MONGODB_URI") or settings.MONGODB_URI
    client = MongoClient(uri, serverSelectionTimeoutMS=8000,
                         connectTimeoutMS=8000, appname="kb_build_readonly")
    client.admin.command("ping")
    return client, ReadOnlyDB(client[settings.MONGODB_DB_NAME])


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------
def clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def get_path(doc: Any, path: str) -> Any:
    cur = doc
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def localized(v: Any) -> dict[str, str] | None:
    """{"EN": "...", "HI": "..."} -> {"en": ..., "hi": ...}; else None."""
    if not isinstance(v, dict):
        return None
    out = {}
    for k, val in v.items():
        lang = _LANG_KEYS.get(str(k).lower())
        if lang and isinstance(val, str) and clean(val):
            out[lang] = clean(val)
    return out or None


def value_for(raw: Any, kind: str, lang: str) -> str | None:
    """Text of `raw` in `lang`, or None if the database has none for it.
    Never translates or invents: a language with no data is simply absent."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, list):
        parts = [value_for(x, kind, lang) for x in raw]
        return ", ".join(p for p in parts if p) or None
    loc = localized(raw)
    if loc is not None:
        return loc.get(lang)
    if isinstance(raw, dict):
        for k in ("title", "name", "label"):
            if k in raw:
                return value_for(raw[k], kind, lang)
        return None
    if isinstance(raw, (int, float)):
        return str(raw) if (kind == "neutral" or lang == "en") else None
    if isinstance(raw, str):
        s = clean(raw)
        if not s:
            return None
        return s if (kind == "neutral" or lang == "en") else None
    return None            # ObjectId, datetime ... never stringify unknown types


def short(s: str, limit: int = 200) -> str:
    first = re.split(r"(?<=[.!?])\s+", s.strip())[0]
    if len(first) > limit:
        first = first[:limit].rsplit(" ", 1)[0]
    return first.rstrip(" .,;:")


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def one_line(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def join_list(items: list[str], lang: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def auto_tags(text: str) -> list[str]:
    low = text.lower()
    return [tag for tag, words in cfg.AUTO_TAGS.items() if any(w in low for w in words)]


# ---------------------------------------------------------------------------
# Facts -> spoken answer + details
# ---------------------------------------------------------------------------
class Resolver:
    """Resolves lists of ids (e.g. hotels.amenitiesIDs) to English names by
    reading the referenced collection. Read-only; results are cached."""

    def __init__(self, db: ReadOnlyDB) -> None:
        self.db = db
        self.cache: dict[tuple[str, str], Any] = {}
        self.stats: dict[str, list[int]] = {}      # collection -> [asked, resolved]

    @staticmethod
    def _oid(i: str):
        try:
            from bson import ObjectId              # installed with pymongo
            return ObjectId(i)
        except Exception:
            return i

    def names(self, collection: str, ids: Any, name_paths: list[str]) -> list[str]:
        ids = [str(i) for i in (ids if isinstance(ids, list) else [ids]) if i]
        need = [i for i in dict.fromkeys(ids) if (collection, i) not in self.cache]
        if need:
            for doc in self.db[collection].find({"_id": {"$in": [self._oid(i) for i in need]}}):
                self.cache[(collection, str(doc["_id"]))] = doc
            for i in need:
                self.cache.setdefault((collection, i), None)
        st = self.stats.setdefault(collection, [0, 0])
        out: list[str] = []
        for i in ids:
            st[0] += 1
            doc = self.cache.get((collection, i))
            if not doc or doc.get("status") not in (None, "active"):
                continue
            for p in name_paths:
                v = value_for(get_path(doc, p), "text", "en")
                if v:
                    out.append(v)
                    st[1] += 1
                    break
        return list(dict.fromkeys(out))


def _fact_from(raw: Any, f: dict, resolver: Resolver | None) -> dict | None:
    kind = f.get("kind", "text")
    if kind == "flag":
        on = raw is True or (isinstance(raw, int) and not isinstance(raw, bool) and raw > 0)
        return {"values": {"en": "Available"}, "items": [f["label"][0].lower()]} if on else None
    if kind == "count":
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
            noun = f["noun"][0] if raw == 1 else f["noun"][1]
            return {"values": {"en": str(raw)}, "items": [f"{raw} {noun}"]}
        return None
    if kind == "lookup":
        if not raw or resolver is None:
            return None
        names = resolver.names(f["collection"], raw, f["name_paths"])
        return {"values": {"en": ", ".join(names)}, "items": names} if names else None
    v = value_for(raw, kind, "en")
    return {"values": {"en": v}, "items": [v]} if v else None


def collect_facts(doc: dict, fields: list[dict], found: set[str],
                  resolver: Resolver | None = None) -> list[dict]:
    facts = []
    for f in fields:
        vis = f.get("visible")
        if vis and get_path(doc, vis) is False:
            continue                                   # hidden on the website
        for p in f["paths"]:
            fact = _fact_from(get_path(doc, p), f, resolver)
            if fact:
                fact["label"] = f["label"]
                fact["details_only"] = bool(f.get("details_only"))
                facts.append(fact)
                found.add(p)
                break
    return facts


def spoken_facts(facts: list[dict]) -> list[dict]:
    return [f for f in facts if not f["details_only"]]


def spoken_from_facts(facts: list[dict], style: str = "facts") -> dict[str, str]:
    facts = spoken_facts(facts)
    if not facts:
        return {}
    if style == "list":
        items = [i for f in facts for i in f["items"]]
        shown = items[:8]
        tail = " and more" if len(items) > 8 else ""
        text = (f"Facilities include {join_list(shown, 'en')}{tail}." if not tail
                else f"Facilities include {', '.join(shown)}{tail}.")
        return {"en": text}
    parts = [f"{f['label'][0]}: {short(f['values']['en'])}." for f in facts[:2]]
    return {"en": " ".join(parts)}


def details_from_facts(facts: list[dict]) -> str:
    return "\n".join(f"- {f['label'][0]}: {f['values']['en']}" for f in facts)



def capacity_sentence(lang: str, title: str, adults: Any, children: Any) -> str | None:
    a = adults if isinstance(adults, int) and not isinstance(adults, bool) else 0
    c = children if isinstance(children, int) and not isinstance(children, bool) else 0
    if a <= 0 and c <= 0:
        return None
    bits = []
    if a:
        bits.append(f"{a} adult{'s' if a != 1 else ''}")
    if c:
        bits.append(f"{c} child{'ren' if c != 1 else ''}")
    return f"The {title} suits up to {' and '.join(bits)}."


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(*, type_: str, title: str, description: str, tags: list[str],
           hotel_id: str, hotel_name: str, spoken: dict[str, str],
           details: str, extra: dict[str, str] | None = None) -> str:
    seen, clean_tags = set(), []
    for t in tags:
        t = one_line(t).replace(",", " ").lower()
        if t and t not in seen:
            seen.add(t)
            clean_tags.append(t)
    lines = [
        "---",
        f"type: {type_}",
        f"title: {one_line(title)}",
        f"description: {one_line(description)}",
        f"tags: [{', '.join(clean_tags)}]",
        f"timestamp: {_TS}",
        f"hotel_id: {hotel_id}",
        f"hotel: {one_line(hotel_name)}",
    ]
    for k, v in (extra or {}).items():
        lines.append(f"{k}: {one_line(v)}")
    lines += ["source: mongodb", "generated: true", "---", "",
              f"# {one_line(title)}", "", "## Spoken answer", ""]
    for lang in LANGS:
        if spoken.get(lang):
            lines += [f"### {lang}", spoken[lang], ""]
    lines += ["## Details", details.strip() or "-", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-hotel lists from their own collections (restaurants, nearbyhotels ...)
# Read-only: find() only, through ReadOnlyDB; BLOCKED collections are refused.
# ---------------------------------------------------------------------------
_ID_LIKE = re.compile(r"^[0-9a-f]{24}$", re.I)
_UNIT_WORDS = {
    "km": "kilometers", "kms": "kilometers", "kilometer": "kilometers",
    "kilometers": "kilometers", "kilometre": "kilometers", "kilometres": "kilometers",
    "m": "meters", "meter": "meters", "meters": "meters", "metre": "meters",
    "metres": "meters", "mi": "miles", "mile": "miles", "miles": "miles",
    "min": "minutes", "mins": "minutes", "minute": "minutes", "minutes": "minutes",
}


def _link_ids(doc: dict, paths: list[str]) -> list[str]:
    """Hotel ids a row points at (ObjectId / string / list / {_id: ..})."""
    for p in paths:
        v = get_path(doc, p)
        out = []
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, dict):
                x = x.get("_id")
            if x is not None and not isinstance(x, bool):
                out.append(str(x))
        if out:
            return out
    return []


def read_distance(doc: dict, spec: dict) -> tuple[str | None, bool]:
    """('3.5 kilometers', False) | (None, True) when the row has a number but
    NO unit anywhere (never guessed) | (None, False) when there is no distance."""
    unit_in_row = ""
    for p in spec.get("distance_unit_paths", []):
        u = get_path(doc, p)
        if isinstance(u, str) and u.strip():
            unit_in_row = u.strip().lower()
            break
    unitless = False
    for path, unit in spec.get("distance", []):
        raw = get_path(doc, path)
        if raw is None or isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            num, word = float(raw), None
        elif isinstance(raw, str):
            m = re.match(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)?", raw)   # "6 kms from Hotel" -> 6, kms
            if not m:
                continue
            num = float(m.group(1))
            w = (m.group(2) or "").lower()
            word = w if w in _UNIT_WORDS else None      # "from", "to" etc. are not units
        else:
            continue
        if num <= 0:
            continue
        unit_txt = word or unit or unit_in_row or cfg.NEARBY_DISTANCE_UNIT
        unit_txt = _UNIT_WORDS.get(str(unit_txt or "").lower())
        if not unit_txt:
            unitless = True
            continue
        n = str(int(num)) if num.is_integer() else f"{num:g}"
        return f"{n} {unit_txt}", False
    return None, unitless


def load_related(db: ReadOnlyDB, spec: dict, hotel_ids: set[str],
                 resolver: Resolver, notes: list[str]) -> dict[str, list[dict]]:
    """{hotel id: [{"name": ..., "facts": [(label, value), ...]}, ...]}"""
    coll = spec["collection"]
    if coll in BLOCKED:
        raise RuntimeError(f"refusing to read '{coll}': customer / booking / credential data")
    docs = list(db[coll].find({}))[:5000]
    found: set[str] = set()
    out: dict[str, list[dict]] = {}
    linked = nameless = unitless = 0
    for doc in docs:
        if doc.get("status") not in (None, "active"):
            continue
        ids = [i for i in _link_ids(doc, spec["hotel_paths"]) if i in hotel_ids]
        if not ids:
            continue
        name = None
        for f in [spec["name"], *spec.get("name_fallbacks", [])]:
            got = collect_facts(doc, [f], found, resolver)
            if got:
                name = got[0]["values"]["en"]
                break
        if not name or _ID_LIKE.match(name):
            nameless += 1
            continue
        facts, seen = [], set()
        for fact in collect_facts(doc, spec["fields"], found, resolver):
            label, val = fact["label"][0], fact["values"]["en"]
            if label in seen or any(_ID_LIKE.match(x) for x in re.split(r",\s*", val)):
                continue                      # duplicate label / a raw id: never spoken
            seen.add(label)
            facts.append((label, val))
        if spec.get("distance"):
            dist, bare = read_distance(doc, spec)
            unitless += bare
            if dist:
                facts.insert(0, ("Distance", dist))
        linked += 1
        for hid in ids:
            out.setdefault(hid, []).append({"name": name, "facts": facts})
    for hid, items in out.items():
        uniq = {i["name"].lower(): i for i in items}          # same name twice -> once
        out[hid] = sorted(uniq.values(), key=lambda i: i["name"].lower())
    notes.append(f"{coll}: {len(docs)} document(s), {linked} linked to an active hotel, "
                 f"{len(out)} hotel(s) get a file")
    if docs and not linked:
        notes.append(f"{coll}: NOTHING linked - none of {spec['hotel_paths']} matched an active "
                     f"hotel id. Run --inspect --collections {coll} and fix hotel_paths / name "
                     f"in kb_config.py")
    if nameless:
        notes.append(f"{coll}: {nameless} row(s) skipped - no readable name")
    if unitless:
        notes.append(f"{coll}: {unitless} row(s) have a distance number with NO unit - skipped; "
                     f"set NEARBY_DISTANCE_UNIT in kb_config.py if they are all in one unit")
    skip = {"_id", "__v", "createdAt", "updatedAt", "status", "icon", "image", "images",
            "createdBy", "updatedBy", "isDeleted"}
    read_top = ({p.split(".")[0] for p in found} | {p.split(".")[0] for p in spec["hotel_paths"]}
                | {p.split(".")[0] for p, _u in spec.get("distance", [])}
                | {p.split(".")[0] for p in spec.get("distance_unit_paths", [])})
    unread = sorted({k for d in docs[:30] for k in d} - skip - read_top)
    if unread:
        notes.append(f"{coll}: fields present but NOT read: {', '.join(unread)} "
                     f"(add them to kb_config.py if they hold names / distances / descriptions)")
    return out


def render_related(spec: dict, items: list[dict], hid: str, hotel_name: str) -> str:
    spoken_items = []
    for it in items:
        d = next((v for k, v in it["facts"] if k == "Distance"), None)
        spoken_items.append(f"{it['name']} at {d}" if d else it["name"])
    shown = spoken_items[:8]
    text = (f"{spec['spoken_lead']} {', '.join(shown)} and more." if len(spoken_items) > 8
            else f"{spec['spoken_lead']} {join_list(shown, 'en')}.")
    lines = []
    for it in items:
        lines.append(f"- {spec['item_label']}: {it['name']}")
        lines += [f"  - {k}: {v}" for k, v in it["facts"]]
    return render(
        type_=spec["type"], title=spec["title"],
        description=f"{spec['title']} for {hotel_name}.",
        tags=list(spec["tags"]) + [it["name"] for it in items],
        hotel_id=hid, hotel_name=hotel_name,
        spoken={"en": text}, details="\n".join(lines),
        extra={"priority": "1"})        # on a tie, beats the older hotel-text file


# ---------------------------------------------------------------------------
# Build (reads Mongo, returns {relative path: text with timestamp placeholder})
# ---------------------------------------------------------------------------
def build_files(db: ReadOnlyDB) -> tuple[dict[str, str], list[str], dict[str, set[str]], Resolver]:
    notes: list[str] = []
    found: dict[str, set[str]] = {t["path"]: set() for t in cfg.TOPICS}
    for k in ("overview", "room", "room-policy"):
        found[k] = set()
    files: dict[str, str] = {}
    resolver = Resolver(db)

    cities = {c["_id"]: c for c in db["cities"].find({"status": "active"})}
    hotels = list(db["hotels"].find({"status": "active"}))
    used_slugs: set[str] = set()
    hotel_ids = {str(h["_id"]) for h in hotels}
    related = {s["collection"]: load_related(db, s, hotel_ids, resolver, notes)
               for s in cfg.RELATED_LISTS}

    for h in hotels:
        hid = str(h["_id"])
        name = clean(str(h.get("name") or ""))
        if not name:
            notes.append(f"hotel {hid}: no name - skipped")
            continue
        city = cities.get(h.get("cityID"))
        if city is None:
            notes.append(f"hotel '{name}': its city is missing/inactive - skipped "
                         f"(the app wouldn't offer it either)")
            continue
        city_name = clean(str(city.get("title") or ""))

        slug = slugify(name) or hid[-8:]
        if slug in used_slugs:
            slug = f"{slug}-{hid[-6:]}"
        used_slugs.add(slug)
        base = f"hotels/{slug}"

        # --- rooms: same rule as ai_tools._get_room_options (must have a rate)
        rooms = list(db["rooms"].find({"hotelIDs": hid, "status": "active"}))
        plan = db["rateplans"].find_one(
            {"hotelId": h["_id"], "status": "active", "isMasterPlan": True})
        priced = set()
        for r in (plan or {}).get("rooms", []):
            if (r.get("prices") or {}).get("guests"):
                priced.add(r.get("roomId"))
        if plan is None:
            notes.append(f"hotel '{name}': no active master rate plan - no room files")
        rooms = [r for r in rooms if r["_id"] in priced]

        room_titles: list[str] = []
        room_detail_lines = []
        used_room_slugs: set[str] = set()
        for room in rooms:
            loc = localized(room.get("title")) or (
                {"en": clean(room["title"])} if isinstance(room.get("title"), str) else {})
            title_en = loc.get("en")
            if not title_en:
                notes.append(f"hotel '{name}': a room has no English title - skipped")
                continue
            room_titles.append(title_en)

            desc = collect_facts(room, cfg.ROOM_DESCRIPTION_FIELDS, found["room"], resolver)
            extra = collect_facts(room, cfg.ROOM_DETAIL_FIELDS, found["room"], resolver)

            parts = []
            cap = capacity_sentence("en", title_en, room.get("adults"), room.get("children"))
            if cap:
                parts.append(cap)
            if desc:
                parts.append(short(desc[0]["values"]["en"]) + ".")
            spoken = {"en": " ".join(parts[:2])} if parts else {}

            lines = [f"- Room type: {title_en}",
                     f"- Max adults per room: {room.get('adults')}",
                     f"- Max children per room: {room.get('children')}"]
            if desc or extra:
                lines.append(details_from_facts(desc + extra))
            lines.append("- Prices are not stored here; they come live from the booking system.")

            rslug = slugify(title_en) or str(room["_id"])[-6:]
            if rslug in used_room_slugs:
                rslug = f"{rslug}-{str(room['_id'])[-6:]}"
            used_room_slugs.add(rslug)

            tags = ["room", "rooms", title_en] + auto_tags(
                " ".join(f["values"]["en"] for f in spoken_facts(desc)))
            files[f"{base}/rooms/{rslug}.md"] = render(
                type_="room", title=title_en,
                description=f"What the {title_en} at {name} offers.",
                tags=tags, hotel_id=hid, hotel_name=name, spoken=spoken,
                details="\n".join(lines), extra={"room_id": str(room["_id"])})

            # per-room cancellation / payment policy file
            pol = collect_facts(room, cfg.ROOM_POLICY_FIELDS, found["room-policy"], resolver)
            if spoken_facts(pol):
                files[f"{base}/rooms/{rslug}-policies.md"] = render(
                    type_="policy", title=f"{title_en} policies",
                    description=f"Cancellation and payment policy for the {title_en} at {name}.",
                    tags=list(cfg.ROOM_POLICY_TAGS) + [title_en],
                    hotel_id=hid, hotel_name=name,
                    spoken=spoken_from_facts(pol), details=details_from_facts(pol),
                    extra={"room_id": str(room["_id"])})

            cap_txt = []
            if room.get("adults"):
                cap_txt.append(f"{room['adults']} adult{'s' if room['adults'] != 1 else ''}")
            if room.get("children"):
                cap_txt.append(f"{room['children']} child{'ren' if room['children'] != 1 else ''}")
            room_detail_lines.append(
                f"- {title_en}" + (f" (up to {', '.join(cap_txt)})" if cap_txt else ""))

        # --- overview
        over = collect_facts(h, cfg.OVERVIEW_FIELDS, found["overview"], resolver)
        star = h.get("starRating")
        htype = clean(str(h.get("hotelType") or "")).lower() or "hotel"
        what = (f"a {star}-star {htype}" if isinstance(star, int) and not isinstance(star, bool)
                and star > 0 else f"a {htype}")
        s1 = f"{name} is {what} in {city_name}."
        n = len(room_titles)
        if n:
            s2 = f"It has {n} room type{'s' if n != 1 else ''}: {join_list(room_titles, 'en')}."
            spoken = {"en": f"{s1} {s2}"}
        else:
            spoken = {"en": s1}
        details = [f"- Hotel: {name}", f"- City: {city_name}"]
        if over:
            details.append(details_from_facts(over))
        if room_detail_lines:
            details.append("- Room types:")
            details += [f"  {l}" for l in room_detail_lines]
        files[f"{base}/overview.md"] = render(
            type_="hotel_overview", title=f"{name} overview",
            description=f"Short introduction to {name} in {city_name} and its room types.",
            tags=["overview", "about", "hotel", name, city_name, "room types"],
            hotel_id=hid, hotel_name=name, spoken=spoken, details="\n".join(details))

        # --- configured topics (only what the DB actually has)
        for t in cfg.TOPICS:
            facts = collect_facts(h, t["fields"], found[t["path"]], resolver)
            if not spoken_facts(facts):
                continue
            style = t.get("spoken_style", "facts")
            spoken_txt = spoken_from_facts(facts, style)
            files[f"{base}/{t['path']}"] = render(
                type_=t["type"], title=t["title"],
                description=f"{t['title']} for {name}.",
                tags=list(t["tags"]) + (auto_tags(spoken_txt.get("en", ""))
                                        if t.get("auto_tags", True) else []),
                hotel_id=hid, hotel_name=name,
                spoken=spoken_txt, details=details_from_facts(facts),
                extra=t.get("extra"))

        # --- lists from their own collections (restaurants, nearbyhotels)
        for spec in cfg.RELATED_LISTS:
            items = related[spec["collection"]].get(hid)
            if items:
                files[f"{base}/{spec['path']}"] = render_related(spec, items, hid, name)

    return files, notes, found, resolver

# ---------------------------------------------------------------------------
# Writing (idempotent) + pruning
# ---------------------------------------------------------------------------
def write_files(out: Path, files: dict[str, str], dry: bool) -> Counter:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stats: Counter = Counter()
    for rel, text in sorted(files.items()):
        path = out / rel
        status = "created"
        if path.exists():
            old = re.sub(r"^timestamp: .*$", f"timestamp: {_TS}",
                         path.read_text(encoding="utf-8"), count=1, flags=re.M)
            if old == text:
                stats["unchanged"] += 1
                continue
            status = "updated"
        stats[status] += 1
        print(f"  {status:9} {rel}")
        if not dry:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text.replace(_TS, today), encoding="utf-8", newline="\n")
    return stats


def find_stale(out: Path, keep: set[str]) -> list[Path]:
    stale = []
    root = out / "hotels"
    if not root.is_dir():
        return stale
    for p in root.rglob("*.md"):
        if p.relative_to(out).as_posix() in keep:
            continue
        if "\ngenerated: true" in p.read_text(encoding="utf-8")[:1000]:
            stale.append(p)
    return stale


# ---------------------------------------------------------------------------
# --inspect
# ---------------------------------------------------------------------------
_SENSITIVE = re.compile(r"pass|token|secret|key|auth|email|phone|mobile|contact", re.I)


def _walk(v: Any, path: str, out: dict) -> None:
    if isinstance(v, dict):
        for k, x in v.items():
            _walk(x, f"{path}.{k}" if path else str(k), out)
    elif isinstance(v, list):
        if not v:
            _record(out, path + "[]", "empty list", None)
        for x in v[:5]:
            _walk(x, path + "[]", out)
    else:
        _record(out, path, type(v).__name__, v)


def _record(out: dict, path: str, tname: str, value: Any) -> None:
    e = out.setdefault(path, {"types": Counter(), "n": 0, "example": None})
    e["types"][tname] += 1
    e["n"] += 1
    if e["example"] is None and value is not None and tname in ("str", "int", "float", "bool"):
        s = str(value)
        if _SENSITIVE.search(path) or "@" in s or re.fullmatch(r"[\d +()-]{8,}", s):
            s = "<masked>"
        e["example"] = s[:40]


# Collections that hold customers, bookings, credentials or logs: never read.
BLOCKED = {
    "access_rights", "admins", "bookingagentdetails", "bookingagents", "bookinglogs",
    "bookings", "booktables", "bookeventandbanquets", "campaigndetails", "campaigns",
    "carts", "chat_session", "chat_sessions", "contactus", "corporatecodes", "coupons",
    "customers", "eventenquiries", "failedbookings", "hotelloginlogs", "hotelmanagers",
    "loginlogs", "nonmembers", "notifications", "otp_verifications", "otprequests",
    "paymentstatuses", "pushtemplates", "roles", "room_selection_locks", "subscriptions",
    "tempbookings", "users", "visitors",
}


def inspect_db(db: ReadOnlyDB, extra: list[str] | None = None) -> None:
    print("Collections in database:", ", ".join(sorted(db.list_collection_names())))
    wanted = ["cities", "hotels", "rooms", "rateplans"]
    for c in extra or []:
        if c in BLOCKED:
            print(f"(refusing to read '{c}': customer / booking / credential data)")
        elif c not in wanted:
            wanted.append(c)
    print(f"(only {', '.join(wanted)} are sampled below; nothing else is ever read)\n")
    for name in wanted:
        coll = db[name]
        total = coll.count_documents({})
        fields: dict = {}
        sampled = 0
        for doc in coll.find({}).limit(30):
            sampled += 1
            _walk(doc, "", fields)
        print(f"=== {name}: {total} document(s), {sampled} sampled ===")
        for path in sorted(fields)[:150]:
            e = fields[path]
            types = "/".join(f"{t}" for t in e["types"])
            ex = f"  e.g. {e['example']!r}" if e["example"] is not None else ""
            print(f"  {path:50} {types:14} in {e['n']}/{sampled}{ex}")
        print()


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: KB_LOCAL_DIR in local mode, "
                         "KB_BUILD_DIR in s3 mode)")
    ap.add_argument("--inspect", action="store_true", help="print Mongo field names and exit")
    ap.add_argument("--collections", default="",
                    help="with --inspect: extra comma-separated collections to sample, "
                         "e.g. amenities,services,wellnesses (customer/booking data is refused)")
    ap.add_argument("--dry-run", action="store_true", help="show changes, write nothing")
    ap.add_argument("--no-publish", action="store_true",
                    help="KB_STORAGE=s3: build only, don't upload")
    ap.add_argument("--force", action="store_true",
                    help="allow deleting more than half of the existing generated files")
    args = ap.parse_args()
    out = args.out or kb_storage.output_dir()
    mode = kb_storage.backend()
    print(f"KB_STORAGE={mode}; working folder: {out}")

    client, db = connect()
    try:
        if args.inspect:
            inspect_db(db, [c.strip() for c in args.collections.split(",") if c.strip()])
            return 0
        files, notes, found, resolver = build_files(db)
    finally:
        client.close()

    kb_storage.pull_current(out)      # s3 mode: start from the published files
    for n in notes:
        print(f"  note: {n}")
    if not files:
        print("ERROR: Mongo returned no usable active hotels - refusing to touch the "
              "knowledge base (wrong database/URI?).")
        return 2

    stale = find_stale(out, set(files))
    existing_generated = len(stale) + sum(1 for r in files if (out / r).exists())
    if stale and len(stale) > max(3, existing_generated // 2) and not args.force:
        print(f"ERROR: would delete {len(stale)} of {existing_generated} generated "
              f"files - that looks wrong. Re-run with --force if it is intended.")
        return 2

    stats = write_files(out, files, args.dry_run)
    for p in stale:
        print(f"  deleted   {p.relative_to(out).as_posix()}")
        if not args.dry_run:
            p.unlink()
    if not args.dry_run:
        for d in sorted((out / "hotels").rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    print(f"\n{len(files)} file(s): {stats['created']} created, {stats['updated']} updated, "
          f"{stats['unchanged']} unchanged, {len(stale)} deleted")
    print("\nTopics with no data in Mongo (not written; fix field names in kb_config.py "
          "after --inspect):")
    any_missing = False
    for t in cfg.TOPICS:
        if not found[t["path"]]:
            any_missing = True
            tried = ", ".join(p for f in t["fields"] for p in f["paths"])
            print(f"  {t['path']}: nothing found (tried {tried})")
    if not any_missing:
        print("  (none)")
    for coll, (asked, got) in resolver.stats.items():
        flag = "" if got else "   <-- no names read: check name_paths (run --inspect --collections " + coll + ")"
        print(f"  lookup {coll}: {got}/{asked} ids resolved to names{flag}")

    if args.dry_run:
        return 0
    kb_storage.write_manifest(out)
    if mode == "s3" and not args.no_publish:
        kb_storage.publish(out)
    elif mode != "s3":
        print(f"files are in {out} (local mode - nothing uploaded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())