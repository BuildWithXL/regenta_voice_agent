"""
okf_store.py - read-only, in-memory hotel knowledge (hotel_kb/).

Step 2 + 3 of the knowledge-base plan:

  * load():  at startup, reads every .md file under hotel_kb/ into memory
             (frontmatter + body). Nothing touches the disk during a call.
  * lookup(): takes a question/topic, returns the 1-3 best snippets, each
             with the short spoken answer in the caller's language.
  * verify_against_mongo(): compares the hotels named in the files with the
             ACTIVE hotels in Mongo, prints warnings for any mismatch, and
             only lets lookup() serve hotels that are active in Mongo.

Like mongo.py, this module never writes anything.

Files come from kb_storage.ensure_local(): the local ./hotel_kb folder in
development (KB_STORAGE=local) or a synced copy of the S3/R2 bucket in
production (KB_STORAGE=s3). Generate them with kb_build.py.

Usage in main.py (once, at startup, after the Mongo name loading):

    import okf_store
    await okf_store.start()                       # load + verify against Mongo
    refresh_task = asyncio.create_task(okf_store.refresh_loop())   # long-running servers

Offline test (no Mongo needed):

    python okf_store.py "do you have breakfast" --lang en --offline
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import kb_storage

KB_DIR = Path(__file__).resolve().parent / "hotel_kb"
LANGS = ("en",)
_REQUIRED_KEYS = ("type", "title", "description", "tags", "timestamp")

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.S)
_H2 = re.compile(r"^## (.+?)\s*$", re.M)
_H3 = re.compile(r"^### (\w+)\s*$", re.M)

# ---------------------------------------------------------------------------
# Synonyms: topic -> phrases (lowercase, substring match against the query).
# A topic only boosts documents that carry the topic name as a tag, so when
# you add a synonym, add it here; when you add a new topic, add the topic
# name to the relevant files' `tags`.
# ---------------------------------------------------------------------------
TOPICS: dict[str, list[str]] = {
    "breakfast": ["breakfast", "naashta", "nashta", "naasta", "morning meal"],
    "check-in": ["check in", "check-in", "checkin", "arrival time"],
    "check-out": ["check out", "check-out", "checkout", "late checkout"],
    "cancellation": ["cancel", "refund", "money back", "non-refundable", "nonrefundable"],
    "wifi": ["wifi", "wi-fi", "wi fi", "internet"],
    "parking": ["parking", "park my car", "park the car", "valet"],
    "pool": ["pool", "swimming"],
    "gym": ["gym", "fitness", "workout"],
    "airport": ["airport", "pickup", "pick up", "pick-up"],
    "directions": ["direction", "how to reach", "how do i reach", "how to get there", "get there", "get to the hotel",
                   "address", "location", "located", "where is the hotel",
                   "where's the hotel", "where are you", "landmark"],
    "nearby": ["nearby", "near the hotel", "attraction", "sightseeing",
               "places to visit", "around the hotel"],
    "child": ["child", "children", "kid", "extra bed", "infant", "baby"],
    "payment": ["payment", "upi", "id proof", "advance payment", "deposit",
                "credit card", "guarantee"],
    "room": ["room", "suite", "deluxe", "standard"],
    "dining": ["restaurant", "dining", "lunch", "dinner", "food", "meals", "cafe"],
    "events": ["banquet", "wedding", "conference", "meeting hall", "events"],
    "wellness": ["spa", "massage", "wellness"],
    "overview": ["about the hotel", "tell me about", "overview", "about this hotel"],
    "amenities": ["amenities", "amenity"],
}


def _phrase_regex(phrase: str) -> re.Pattern:
    """Whole-word match for short phrases ('spa' must not match 'space',
    'room' must not match 'bathroom'); prefix match for longer ones
    ('cancel' matches 'cancellation')."""
    body = re.escape(phrase)
    if len(phrase) <= 4:
        return re.compile(rf"(?<![a-z0-9]){body}(?:s|es)?(?![a-z0-9])")
    return re.compile(rf"(?<![a-z0-9]){body}")


_TOPIC_RE = {t: [_phrase_regex(p) for p in phrases] for t, phrases in TOPICS.items()}

_STOP = {
    "the", "is", "are", "a", "an", "do", "you", "have", "what", "which", "of", "to", "in", "on", "at", "for", "and", "or", "me", "my", "i", "can", "does", "there", "it", "with", "hai", "hain", "kya",
}


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------
@dataclass
class KBDoc:
    path: str                      # relative to hotel_kb/
    type: str
    title: str
    description: str
    tags: list[str]
    timestamp: str
    hotel_id: str | None           # Mongo hotels._id as a string; None = general
    hotel_slug: str | None         # folder name under hotels/
    spoken: dict[str, str]         # lang -> two-sentence answer
    details: str                   # detailed notes (below the spoken answer)
    meta: dict[str, str] = field(default_factory=dict)   # extra frontmatter (room_id ...)
    # precomputed for fast matching
    tag_set: set[str] = field(default_factory=set, repr=False)
    tag_tokens: set[str] = field(default_factory=set, repr=False)
    title_tokens: set[str] = field(default_factory=set, repr=False)
    desc_tokens: set[str] = field(default_factory=set, repr=False)


@dataclass
class Snippet:
    title: str
    type: str
    hotel_id: str | None
    path: str
    spoken: str                    # in the requested language (falls back to en)
    details: str
    score: float
    meta: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    """Unicode-aware word split. Keeps combining marks (Hindi/Gujarati
    matras), which a plain \\w regex would split on."""
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if unicodedata.category(ch)[0] in "LMN":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def _norm(name: str) -> str:
    """Lowercase + collapse every kind of whitespace (hotel names in Mongo can
    contain narrow no-break spaces, U+202F)."""
    return re.sub(r"\s+", " ", name).strip().lower()


def _useful_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if len(t) > 1 and t not in _STOP}


def _parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        items = [x.strip().strip("'\"") for x in raw[1:-1].split(",")]
        return [i for i in items if i]
    return raw.strip("'\"")


def _parse_frontmatter(text: str) -> tuple[dict, str] | None:
    m = _FRONTMATTER.match(text)
    if not m:
        return None
    meta: dict = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = _parse_value(value)
    return meta, m.group(2)


def _split_h2(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(_H2.finditer(body))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[m.group(1).strip().lower()] = body[m.end():end].strip()
    return sections


def _parse_spoken(block: str) -> dict[str, str]:
    parts = _H3.split(block)          # ['', 'en', text, 'hi', text, ...]
    return {
        parts[i].lower(): " ".join(parts[i + 1].split())
        for i in range(1, len(parts) - 1, 2)
        if parts[i].lower() in LANGS
    }


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
class KnowledgeStore:
    def __init__(self) -> None:
        self.docs: list[KBDoc] = []
        self.load_warnings: list[str] = []
        # hotel_id -> hotel name, for hotels ACTIVE in Mongo. None until
        # verify_against_mongo() (or allow_all() in tests) has run: until
        # then no hotel-specific knowledge is served.
        self._active: dict[str, str] | None = None
        self._id_by_name: dict[str, str] = {}
        self._loaded_hash: str | None = None
        self._last_warn: tuple | None = None

    # -- Step 2: load -------------------------------------------------------
    def load(self, kb_dir: Path = KB_DIR) -> list[str]:
        kb_dir = Path(kb_dir)
        self.load_warnings = []
        if not kb_dir.is_dir():
            self.docs = []
            self.load_warnings.append(
                f"[kb] folder not found: {kb_dir} - run `python kb_build.py` first")
            return self.load_warnings

        docs: list[KBDoc] = []
        for path in sorted(kb_dir.rglob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            doc = self._parse_file(path, kb_dir)
            if doc is not None:
                docs.append(doc)
        self.docs = docs               # swapped in one step
        self._loaded_hash = kb_storage.manifest_hash(kb_dir)
        return self.load_warnings

    def _warn(self, msg: str) -> None:
        self.load_warnings.append(msg)

    def _parse_file(self, path: Path, root: Path) -> KBDoc | None:
        rel = path.relative_to(root)
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        parsed = _parse_frontmatter(text)
        if parsed is None:
            self._warn(f"[kb] {rel}: no frontmatter - skipped")
            return None
        meta, body = parsed

        missing = [k for k in _REQUIRED_KEYS if not meta.get(k)]
        if missing:
            self._warn(f"[kb] {rel}: missing frontmatter {missing} - skipped")
            return None

        parts = rel.parts
        slug = parts[1] if parts[0] == "hotels" and len(parts) > 2 else None
        hotel_id = str(meta.get("hotel_id") or "").strip() or None
        if slug and not hotel_id:
            self._warn(f"[kb] {rel}: lives under hotels/{slug}/ but has no "
                       f"hotel_id - skipped")
            return None

        sections = _split_h2(body)
        spoken = _parse_spoken(sections.get("spoken answer", ""))
        if not spoken.get("en"):
            self._warn(f"[kb] {rel}: no English spoken answer")
        details = sections.get("details", "")

        tags = meta["tags"] if isinstance(meta["tags"], list) else [meta["tags"]]
        tags = [t.lower() for t in tags]
        doc = KBDoc(
            path=str(rel), type=str(meta["type"]), title=str(meta["title"]),
            description=str(meta["description"]), tags=tags,
            timestamp=str(meta["timestamp"]), hotel_id=hotel_id,
            hotel_slug=slug, spoken=spoken, details=details,
            meta={k: str(v) for k, v in meta.items()
                  if k not in _REQUIRED_KEYS and k != "hotel_id"
                  and not isinstance(v, list)},
        )
        doc.tag_set = set(tags)
        doc.tag_tokens = set().union(*(_useful_tokens(t) for t in tags)) if tags else set()
        doc.title_tokens = _useful_tokens(doc.title)
        doc.desc_tokens = _useful_tokens(doc.description)
        return doc

    # -- Step 3: consistency check against Mongo ----------------------------
    def set_active(self, active_hotels: dict[str, str]) -> None:
        """active_hotels: {hotel _id as str: hotel name}."""
        self._active = dict(active_hotels)
        self._id_by_name = {_norm(n): i for i, n in self._active.items()}

    def allow_all(self) -> None:
        """Testing only: treat every hotel named in the files as active."""
        self.set_active({d.hotel_id: d.hotel_slug or d.hotel_id
                         for d in self.docs if d.hotel_id})

    def check_consistency(self) -> list[str]:
        """Compares files vs the active hotels last given to set_active()."""
        active = self._active or {}
        files_by_hotel: dict[str, list[KBDoc]] = {}
        for d in self.docs:
            if d.hotel_id:
                files_by_hotel.setdefault(d.hotel_id, []).append(d)

        warnings: list[str] = []
        for hid, docs in files_by_hotel.items():
            if hid not in active:
                warnings.append(
                    f"[kb] hotels/{docs[0].hotel_slug}: {len(docs)} file(s) "
                    f"name hotel_id '{hid}', which is missing or inactive in "
                    f"Mongo - this knowledge will NOT be served")
        for hid, name in active.items():
            if hid not in files_by_hotel:
                warnings.append(
                    f"[kb] active hotel '{name}' ({hid}) has no knowledge files")
        return warnings

    def _print_once(self, warnings: list[str], quiet: bool) -> None:
        key = tuple(warnings)
        if quiet and key == self._last_warn:
            return                      # background refresh: only report changes
        self._last_warn = key
        for w in warnings:
            print(f"  {w}")

    async def verify_against_mongo(self, quiet: bool = False) -> list[str]:
        import mongo   # imported here so parsing/lookup work without motor
        try:
            active = await mongo.get_active_hotels()
        except Exception as exc:
            msg = f"[kb] couldn't read active hotels from Mongo ({exc})"
            if quiet and self._active is not None:
                msg += "; keeping the last known active-hotel list"
            else:
                # Fail closed: if we can't confirm which hotels are active,
                # serve no hotel-specific knowledge.
                self.set_active({})
                msg += "; hotel knowledge is DISABLED until Mongo is reachable"
            self._print_once([msg], quiet)
            return [msg]

        self.set_active(active)
        warnings = self.check_consistency()
        self._print_once(warnings, quiet)
        if not quiet:
            serving = len({d.hotel_id for d in self.docs if d.hotel_id in active})
            print(f"  [kb] loaded {len(self.docs)} file(s); "
                  f"serving {serving} of {len(active)} active hotel(s)")
        return warnings

    # -- startup + background refresh ---------------------------------------
    async def start(self) -> None:
        """Fetch the files (local dir or storage bucket), load, verify."""
        kb_dir = await asyncio.to_thread(kb_storage.ensure_local)
        for w in self.load(kb_dir):
            print(f"  {w}")
        await self.verify_against_mongo()

    async def refresh_loop(self, interval: float | None = None) -> None:
        """Every KB_REFRESH_SECONDS (default 300): pick up newly published
        files and re-check which hotels are active in Mongo. A failure
        keeps the last good copy."""
        interval = interval or float(os.getenv("KB_REFRESH_SECONDS", "300"))
        while True:
            await asyncio.sleep(interval)
            try:
                kb_dir = await asyncio.to_thread(kb_storage.ensure_local)
                if kb_storage.manifest_hash(kb_dir) != self._loaded_hash:
                    for w in self.load(kb_dir):
                        print(f"  {w}")
                    print(f"  [kb] knowledge files updated ({len(self.docs)} file(s))")
                await self.verify_against_mongo(quiet=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"  [kb] refresh failed ({exc}); keeping the last good copy")

    # -- Step 2: lookup -----------------------------------------------------
    def _resolve_hotel(self, hotel: str | None) -> str | None:
        """Accepts a hotel name (as in session.checklist.hotel) or an id."""
        if not hotel:
            return None
        h = hotel.strip()
        return self._id_by_name.get(_norm(h)) or (h if self._active and h in self._active else "")

    def topics_in(self, query: str) -> set[str]:
        """Which knowledge topics (breakfast, gym, cancellation ...) the
        text mentions."""
        q = (query or "").lower()
        return {t for t, pats in _TOPIC_RE.items() if any(p.search(q) for p in pats)}

    def hotel_in(self, text: str) -> str | None:
        """Id of an ACTIVE hotel whose full name appears in `text`
        (longest name wins), else None."""
        t = _norm(text or "")
        best: tuple[int, str] | None = None
        for name, hid in self._id_by_name.items():
            if name and name in t and (best is None or len(name) > best[0]):
                best = (len(name), hid)
        return best[1] if best else None

    def docs_for_hotel(self, hotel: str | None) -> list[KBDoc]:
        """Every file of ONE hotel (name or id), only if it is active in
        Mongo. Read-only; [] for an unknown / inactive hotel."""
        hid = self._resolve_hotel(hotel)
        if not hid or self._active is None or hid not in self._active:
            return []
        return [d for d in self.docs if d.hotel_id == hid]

    def lookup(self, query: str, hotel: str | None = None,
               lang: str = "en", limit: int = 3,
               require_topic: bool = False) -> list[Snippet]:
        """Return the 1-3 best snippets for a question or topic.

        hotel: hotel name or Mongo id to restrict to (recommended once the
               caller has picked a hotel). Hotel files are only ever served
               for hotels that are active in Mongo.
        require_topic: only return files whose tags match a topic named in
               the query (high precision - used by the voice agent so an
               ordinary booking answer is never mistaken for a question).
        """
        q = (query or "").lower()
        if not q.strip():
            return []
        limit = max(1, min(limit, 3))

        hotel_id = self._resolve_hotel(hotel)
        if hotel is not None and not hotel_id:
            return []                      # unknown / inactive hotel

        topics = self.topics_in(q)
        qtokens = _useful_tokens(q)

        scored: list[tuple[float, KBDoc]] = []
        for d in self.docs:
            if d.hotel_id is not None:
                if self._active is None or d.hotel_id not in self._active:
                    continue               # not (known to be) active in Mongo
                if hotel_id and d.hotel_id != hotel_id:
                    continue
            if require_topic and not (topics & d.tag_set):
                continue
            score = (5 * len(topics & d.tag_set)
                     + 2 * len(qtokens & d.tag_tokens)
                     + 2 * len(qtokens & d.title_tokens)
                     + 1 * len(qtokens & d.desc_tokens))
            if score > 0:
                scored.append((score, d))

        def _rank(x):                  # score first; on a tie a file with `priority: 1` wins
            try:
                return (x[0], int(x[1].meta.get("priority", "0")))
            except ValueError:
                return (x[0], 0)
        scored.sort(key=_rank, reverse=True)
        if not scored:
            return []
        cutoff = scored[0][0] * 0.4        # drop weak trailing matches
        return [
            Snippet(
                title=d.title, type=d.type, hotel_id=d.hotel_id, path=d.path,
                spoken=d.spoken.get(lang) or d.spoken.get("en", ""),
                details=d.details, score=s, meta=d.meta,
            )
            for s, d in scored[:limit] if s >= cutoff
        ]


# Single shared instance, same pattern as session.store / tts_client._client.
store = KnowledgeStore()


def load(kb_dir: Path = KB_DIR) -> list[str]:
    warnings = store.load(kb_dir)
    for w in warnings:
        print(f"  {w}")
    return warnings


async def verify_against_mongo(quiet: bool = False) -> list[str]:
    return await store.verify_against_mongo(quiet=quiet)


async def start() -> None:
    await store.start()


async def refresh_loop(interval: float | None = None) -> None:
    await store.refresh_loop(interval)


def lookup(query: str, hotel: str | None = None, lang: str = "en",
           limit: int = 3, require_topic: bool = False) -> list[Snippet]:
    return store.lookup(query, hotel=hotel, lang=lang, limit=limit,
                        require_topic=require_topic)


def topics_in(query: str) -> set[str]:
    return store.topics_in(query)


def hotel_in(text: str) -> str | None:
    return store.hotel_in(text)


def hotel_docs(hotel: str | None) -> list[KBDoc]:
    return store.docs_for_hotel(hotel)


def enabled() -> bool:
    """Kill switch: KB_ANSWERS=off stops the voice agent using the knowledge
    base (nothing else changes)."""
    return os.getenv("KB_ANSWERS", "on").strip().lower() not in ("off", "0", "false", "no")


def ready() -> bool:
    return enabled() and bool(store.docs)


# ---------------------------------------------------------------------------
# Tiny CLI for testing:  python okf_store.py "breakfast" --lang hi --offline
# ---------------------------------------------------------------------------
async def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--lang", default="en", choices=LANGS)
    ap.add_argument("--hotel", default=None)
    ap.add_argument("--offline", action="store_true",
                    help="skip Mongo; treat every hotel in the files as active")
    args = ap.parse_args()

    load()
    if args.offline:
        store.allow_all()
    else:
        await verify_against_mongo()
        import mongo
        await mongo.close()

    results = lookup(args.query, hotel=args.hotel, lang=args.lang)
    if not results:
        print("(no match)")
    for r in results:
        print(f"\n[{r.score:.0f}] {r.title}  ({r.path})\n  {r.spoken}")


if __name__ == "__main__":
    asyncio.run(_cli())