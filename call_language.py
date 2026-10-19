"""
Call language: the language choice at the start of the call, switching
mid-call, and the bridge between the caller's language and the call flow
(which works in English).

  * Languages come from languages.json (config.LANGUAGES): name, STT code and
    mode, TTS code, voice, aliases, fixed lines. A new language is config only;
    a fixed line it lacks is translated from the English one.
  * Until a language is chosen, STT runs in PHONE_STT_LANG ("auto"). After that
    session_stt() reconnects Sarvam STT with that language's code/mode. With
    mode "translate" Sarvam returns English, so the flow reads it as is;
    anything still in another script is translated to English with Groq.
  * Replies are written in English by the flow and translated here (Groq), with
    hotel / restaurant / hall names and phone numbers held back as tokens so
    they reach the caller exactly as stored.
  * A mid-call switch needs an explicit request ("please speak English"):
    English words inside a Hindi reply (Hinglish) never switch the language.
"""
from __future__ import annotations

import re
from collections import OrderedDict

from groq import AsyncGroq

import sarvam_stt_client
from config import LANGUAGES, require, settings

_LANGS: dict = LANGUAGES["languages"]
DEFAULT = LANGUAGES.get("default", "en")
_MODEL = LANGUAGES.get("translate_model") or settings.GROQ_MODEL
_MARKERS = re.compile(r"^(\s*\[[^\]]*\]\s*)+")
_FILLERS = {"hmm", "hm", "um", "umm", "uh", "uhh", "ah", "er", "huh", "mm", "hello", "hi", "hey",
            "haan", "ha", "ji", "ok", "okay", "yes", "no", "yeah", "the", "a", "please", "sorry"}
_SWITCH_CUES = re.compile(r"\b(speak|talk|continue|reply|respond|answer|converse|switch|change|use|"
                          r"prefer|language|baat|bol\w*|bhasha|bhasa)\b")
_OTHERS_SPEAK = re.compile(r"\b(staff|they|people|employees?|waiters?|chefs?|guides?|anyone|someone|somebody|"
                           r"newspapers?|menus?|channels?|tv|movies?)\b")   # "do the staff speak Gujarati?"
_ASCII = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
                        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " "})
_client: AsyncGroq | None = None
_cache: "OrderedDict[tuple[str, str], str]" = OrderedDict()


# ------------------------------------------------------------- config
def key_of(session) -> str:
    return session.call_language or DEFAULT


def _cfg(key: str) -> dict:
    return _LANGS.get(key) or _LANGS[DEFAULT]


def tts_code(key: str | None) -> str | None:
    """'hi' -> 'hi-IN' (tts_client)."""
    return _LANGS[key].get("tts_code") if key in _LANGS else None


def voice(key: str | None) -> str | None:
    return (_LANGS.get(key) or {}).get("voice") or None


def stt_settings(session) -> tuple[str, str | None]:
    """(language code, mode) Sarvam STT should run with for this call right now."""
    if not session.call_language:
        return settings.PHONE_STT_LANG, None
    cfg = _cfg(session.call_language)
    return cfg.get("stt_code") or "en-IN", cfg.get("stt_mode") or None


async def line(session, name: str) -> str:
    """A fixed line in the call's language (translated from English if missing)."""
    text = _cfg(key_of(session))["lines"].get(name)
    return text if text else await localize(session, _LANGS[DEFAULT]["lines"][name])


def _english_line(name: str) -> str:
    return _LANGS[DEFAULT]["lines"][name]


# ------------------------------------------------------------- STT
def session_stt(session, api_key: str, audio, start=None):
    """STT results for the call; reconnects Sarvam whenever the call's language
    (and so the STT code/mode) changes, before reading the next utterance."""
    start = start or sarvam_stt_client.stream_transcribe
    while not audio.closed:
        current = stt_settings(session)
        print(f"  [lang] STT language={current[0]} mode={current[1] or 'default'}")
        source = start(api_key, current[0], audio.generator(), mode=current[1])
        try:
            while stt_settings(session) == current:
                try:
                    result = next(source)
                except StopIteration:
                    return
                yield result
        finally:
            source.close()


# ------------------------------------------------------------- which language
def _named(text: str) -> str | None:
    """Language the caller named ('Hindi', 'हिंदी', 'Gujarati mein'); the last one wins."""
    low, best, at = text.lower(), None, -1
    for key, cfg in _LANGS.items():
        for alias in cfg.get("aliases", []) + [cfg["name"]]:
            a = alias.lower()
            m = (list(re.finditer(rf"\b{re.escape(a)}\b", low)) if a.isascii()
                 else [x for x in re.finditer(re.escape(a), low)])
            if m and m[-1].start() > at:
                best, at = key, m[-1].start()
    return best


def _script_language(text: str) -> str | None:
    counts = {k: len(re.findall(f"[{c['script_range']}]", text))
              for k, c in _LANGS.items() if c.get("script_range")}
    key = max(counts, key=counts.get) if counts else None
    return key if key and counts[key] else None


def _english_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 1 and w not in _FILLERS]


def _choose(session, key: str) -> None:
    session.call_language = key
    session.language = key
    print(f"  [lang] call language -> {_cfg(key)['name']}")


async def choose(session, user_message: str, route) -> str:
    """The caller's answer to 'which language?'. route(session, english) runs the flow."""
    text = _MARKERS.sub("", user_message).strip()
    named = _named(text)
    if named:                                         # "Hindi" / "ગુજરાતી" / "Gujarati mein"
        _choose(session, named)
        session.add_message("user", user_message)
        session.add_message("assistant", f"{_english_line('chosen')} {_english_line('ask_city')}")
        return f"{await line(session, 'chosen')} {await line(session, 'ask_city')}"
    implied = _script_language(text) or ("en" if _english_words(text) else None)
    if implied:                                       # they just answered in a language
        _choose(session, implied)
        reply = await route(session, await to_english(session, user_message))
        return f"{await line(session, 'chosen')} {await localize(session, reply)}"
    session.language_attempts += 1
    session.add_message("user", user_message)
    if session.language_attempts < 2:                 # ask once more, in English and Hindi
        ask = LANGUAGES["ask_again"]
        session.language = ask.get("lang", DEFAULT)
        session.add_message("assistant", ask["text"])
        return ask["text"]
    _choose(session, DEFAULT)                         # still unclear: English
    english = f"{_english_line('fallback')} {_english_line('ask_city')}"
    session.add_message("assistant", english)
    return english


async def switch_request(session, user_message: str, english: str) -> str | None:
    """'Please speak English' / 'हिंदी में बात करो' -> switch now and repeat the last question.
    english: the caller's words in English (to_english)."""
    text = _MARKERS.sub("", user_message).strip()
    target = _named(text) or _named(english)
    if not target:
        return None
    low = _MARKERS.sub("", english).lower()
    rest = [w for w in _english_words(low) if w not in ("in", "mein", "me", "only", "language")
            and _named(w) is None]
    if _OTHERS_SPEAK.search(low) or not (_SWITCH_CUES.search(low) or not rest):
        return None                                   # "is the menu in English?" is not a request
    _choose(session, target)
    session.add_message("user", user_message)
    last = next((m["content"] for m in reversed(session.conversation[:-1])
                 if m["role"] == "assistant"), "")
    question = re.findall(r"[^.!?।]*\?", last)
    again = question[-1].strip() if question else _english_line("ask_city")
    session.add_message("assistant", f"{_english_line('chosen')} {again}")
    return f"{await line(session, 'chosen')} {await localize(session, again)}"


# ------------------------------------------------------------- translation
def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=require("GROQ_API_KEY"))
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def _complete(system: str, text: str) -> str:
    response = await _get_client().chat.completions.create(
        model=_MODEL, temperature=0.2, max_tokens=800, reasoning_effort="low",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text}])
    return (response.choices[0].message.content or "").strip()


async def to_english(session, user_message: str) -> str:
    """The caller's words in English for the flow (markers like [transcription
    uncertain] kept). English calls and Sarvam 'translate' output pass straight through."""
    if not session.call_language or session.call_language == DEFAULT:
        return user_message
    marks = _MARKERS.match(user_message)
    prefix, text = (marks.group(0) if marks else ""), _MARKERS.sub("", user_message).strip()
    cfg = _cfg(session.call_language)
    if cfg.get("stt_mode") == "translate" and not _script_language(text):
        return user_message
    try:
        english = await _complete(
            "Translate what a hotel caller said into plain English. Output ONLY the English. Write hotel, "
            "restaurant, city and place names in Latin letters the way they are normally spelled "
            "(e.g. 'बेंगलुरु' -> 'Bengaluru'). Keep numbers as digits.", text)
    except Exception as exc:
        print(f"  [lang] could not translate the caller's words: {exc!r}")
        return user_message
    english = english.translate(_ASCII)          # 'check‑in' (U+2011) must match the flow's 'check-in'
    print(f"  [lang] caller (en): {english}")
    return prefix + english


def _protect(session, english: str) -> tuple[str, dict[str, str]]:
    """Hotel / restaurant / hall names and phone numbers -> ⟦HOTEL1⟧ ..."""
    st = session.regenta or {}
    names: dict[str, str] = {}
    for h in st.get("hotels") or []:
        full = " ".join(h["name"].split())
        names[full] = "HOTEL"
        parts = [p.strip() for p in full.split(",")]
        names[", ".join(parts[:-1]) if len(parts) > 1 else full] = "HOTEL"
    for details in (st.get("details") or {}).values():
        for r in details.get("restaurants") or []:
            if r.get("name"):
                names[" ".join(str(r["name"]).split())] = "RESTAURANT"
        for hall in details.get("banquet_halls") or []:
            if hall.get("title"):
                names[hall["title"]] = "HALL"
        for title in (details.get("wellnesses") or []) + (details.get("services") or []):
            names.setdefault(title, "TREATMENT")       # 'Aroma Therapy', 'Heated Swimming Pool'
    names = {n: k for n, k in names.items() if n.lower() not in ("spa", "restaurant", "swimming pool")}
    tokens: dict[str, str] = {}
    counts: dict[str, int] = {}

    def token(kind: str, value: str) -> str:
        counts[kind] = counts.get(kind, 0) + 1
        tok = f"⟦{kind}{counts[kind]}⟧"
        tokens[tok] = value
        return tok

    text = re.sub(r"\b\d(?: \d){3,}(?:, \d(?: \d)+)?", lambda m: token("PHONE", m.group(0)), english)
    for name in sorted((n for n in names if len(n) > 2), key=len, reverse=True):
        if name in text:
            text = re.sub(rf"(?<![\w⟦]){re.escape(name)}(?!\w)", lambda m, n=name: token(names[n], n), text)
    return text, tokens


def _problems(out: str, tokens: dict[str, str], cfg: dict) -> list[str]:
    bad = [f"keep {t} exactly once" for t in tokens if out.count(t) != 1]
    if cfg.get("script_range"):
        body = re.sub(r"⟦[A-Z]+\d+⟧", "", out)
        native = len(re.findall(f"[{cfg['script_range']}]", body))
        latin = len(re.findall(r"[A-Za-z]", body))
        if native < 0.6 * (native + latin):
            bad.append(f"write it in {cfg.get('script', cfg['name'])} script, not English")
    return bad


async def localize(session, english: str) -> str:
    """The flow's English reply in the call's language (English calls: unchanged)."""
    key = key_of(session)
    if key == DEFAULT or not english.strip():
        return english
    cfg = _cfg(key)
    text, tokens = _protect(session, english)
    cached = _cache.get((key, text))
    if cached is None:
        system = (f"Translate the hotel phone assistant's reply from English into {cfg['name']}; it is read "
                  f"aloud by text-to-speech. Output ONLY the {cfg['name']} text, in {cfg.get('script', '')} script.\n"
                  f"- Natural, polite, short spoken {cfg['name']}, like a hotel receptionist in India.\n"
                  "- Translate every sentence and every piece of information; add nothing, drop nothing.\n"
                  "- Tokens like ⟦HOTEL1⟧, ⟦RESTAURANT2⟧, ⟦HALL1⟧, ⟦TREATMENT1⟧, ⟦PHONE1⟧ stand for a hotel name, "
                  "restaurant name, banquet hall name, spa treatment / service name or phone number: copy each "
                  "one exactly, once, and build the sentence around it.\n- Keep room-type names as they are (not translated).\n"
                  f"- Keep numbers as digits.\n- {cfg.get('style', '')}")
        out, bad = "", []
        for attempt in range(2):
            try:
                out = await _complete(system, text if not attempt else
                                      f"{text}\n\n(Your last answer broke these rules: {'; '.join(bad)})")
            except Exception as exc:
                print(f"  [lang] translation failed: {exc!r}")
                out = ""
                break
            bad = _problems(out, tokens, cfg)
            if not bad:
                break
            print(f"  [lang] translation retry ({'; '.join(bad)})")
        if not out or any(out.count(t) != 1 for t in tokens):
            print(f"  [lang] WARNING: no usable {cfg['name']} translation; speaking English")
            return english
        cached = out
        _cache[(key, text)] = out
        if len(_cache) > 300:
            _cache.popitem(last=False)
    for tok, value in tokens.items():
        cached = cached.replace(tok, value)
    return cached
