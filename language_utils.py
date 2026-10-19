def detect_script_language(text: str, default: str = "en") -> str:
    gu = sum(1 for c in text if "\u0A80" <= c <= "\u0AFF")   # Gujarati block
    hi = sum(1 for c in text if "\u0900" <= c <= "\u097F")   # Devanagari block
    if gu == 0 and hi == 0:
        return default        # Latin script (incl. romanized Hinglish) -> default
    return "gu" if gu >= hi else "hi"
import re

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# 3+ single letters in a row: "D R A S H T I" / "d, r, a, s, h" -> "DRASHTI"
_LETTER_SEQ = re.compile(r"\b(?:[A-Za-z][\s,.-]+){2,}[A-Za-z]\b")

_SYMBOL_MAP = [
    (r"\bat the rate( of)?\b", "@"),
    (r"\bdot\b", "."),
    (r"एट\s*द\s*रेट", "@"),
    (r"डॉट", "."),
]

def detect_language(text: str) -> str:
    """Devanagari script means Hindi. Romanized Hindi looks like English
    here, but the LLM reads the actual words, so only the TTS voice
    choice is affected."""
    return "hi" if _DEVANAGARI.search(text) else "en"

def collapse_spelled_letters(text: str) -> str:
    """'D R A S H T I' -> 'DRASHTI' (for spelled names and emails)."""
    return _LETTER_SEQ.sub(
        lambda m: re.sub(r"[\s,.-]+", "", m.group(0)), text
    )

def normalize_spoken_symbols(text: str) -> str:
    text = re.sub(r"\bunderscore\b", "_", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(dash|hyphen)\b", "-", text, flags=re.IGNORECASE)
    # A bare "at" becomes "@" only when a spoken domain follows it
    # ("drashti at gmail dot com"), so ordinary sentences with "at" are safe.
    text = re.sub(
        r"\s+\bat\b\s+(?=[\w-]+(?:\s+dot\s+[\w-]+)+)",
        "@", text, flags=re.IGNORECASE,
    )
    for pattern, repl in _SYMBOL_MAP:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\s*@\s*", "@", text)  # "drashti @ gmail" -> "drashti@gmail"
    prev = None
    while prev != text:  # "gmail . co . in" -> "gmail.co.in"
        prev = text
        text = re.sub(r"(@[\w.-]+)\s*\.\s*(\w)", r"\1.\2", text)
    return text

# ---------------------------------------------------------------- booking IDs
_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "to": "2", "too": "2",
    "three": "3", "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "ate": "8", "nine": "9",
    "शून्य": "0", "ज़ीरो": "0", "जीरो": "0", "एक": "1", "दो": "2", "तीन": "3",
    "चार": "4", "पांच": "5", "पाँच": "5", "छह": "6", "छः": "6", "सात": "7",
    "आठ": "8", "नौ": "9",
}
_LETTER_NAMES = {
    "ay": "a", "bee": "b", "be": "b", "cee": "c", "see": "c", "sea": "c", "dee": "d",
    "ee": "e", "eff": "f", "gee": "g", "aitch": "h", "eye": "i", "jay": "j",
    "kay": "k", "ell": "l", "el": "l", "em": "m", "en": "n", "pea": "p", "pee": "p",
    "queue": "q", "cue": "q", "are": "r", "ar": "r", "ess": "s", "tea": "t", "tee": "t",
    "you": "u", "vee": "v", "ex": "x", "why": "y", "wy": "y", "zed": "z", "zee": "z",
}
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_MULT = {"double": 2, "triple": 3, "डबल": 2, "ट्रिपल": 3}
_BK_START = re.compile(r"\b(?:b[\s.,-]*k|bee\s+kay|bk)\b", re.I)


def _id_token(tok: str) -> str | None:
    """One spoken token -> its ID characters, or None if it is not part of an ID."""
    t = tok.lower().strip(".,;:!?")
    if not t:
        return ""
    if t in ("dash", "hyphen", "-", "and", "then"):
        return ""
    t = t.translate(_DEVANAGARI_DIGITS)
    if t in _DIGIT_WORDS:
        return _DIGIT_WORDS[t]
    if t in _LETTER_NAMES:
        return _LETTER_NAMES[t]
    if re.fullmatch(r"[a-z0-9]+", t):
        return t
    return None


def normalize_spoken_booking_id(text: str) -> str:
    """'bk seven bee double eight four eight five e' -> 'BK-7B88485E'.
    Only the run of tokens after a 'BK' marker is rewritten."""
    m = _BK_START.search(text)
    if not m:
        return text
    tokens = text[m.end():].split()
    chars, used, mult = "", 0, 1
    for tok in tokens:
        if len(chars) >= 8:
            break
        low = tok.lower().strip(".,;:!?")
        if low in _MULT:
            mult, used = _MULT[low], used + 1
            continue
        piece = _id_token(tok)
        if piece is None:
            break
        if mult > 1 and piece:
            piece = piece[0] * mult
        mult = 1
        if len(chars) + len(piece) > 8:
            break
        chars += piece
        used += 1
    if len(chars) < 8:
        return text  # incomplete: leave the transcript as heard
    rest = " ".join(tokens[used:])
    return f"{text[:m.start()]}BK-{chars.upper()} {rest}".rstrip()
