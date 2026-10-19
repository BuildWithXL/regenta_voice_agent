"""
Deepgram Flux streaming STT client -- the actual call to Deepgram.

Sends audio chunks as they arrive (from mic or file) and yields back
TranscriptResult objects as Flux responds -- interim updates first, then a
final result each time it detects the speaker's turn ended.

IMPORTANT: Flux is a newer Deepgram product and its SDK surface may shift as
it matures. Before relying on this in production, sanity-check the message
attribute names (message.type, message.transcript, message.words) and the
turn-ending control messages against the current docs:
https://developers.deepgram.com/docs/flux/quickstart

Setup:
    pip install deepgram-sdk python-dotenv
    Get an API key at https://console.deepgram.com -> set DEEPGRAM_API_KEY
"""

import queue
import threading
import time
from dataclasses import dataclass, field

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from config import settings
from language_utils import detect_script_language

@dataclass
class TranscriptResult:
    """Shared data shape returned by this STT backend, so the rest of
    the pipeline (main.py) doesn't need to know Deepgram-specific
    details. Was its own transcript_result.py file; merged in here
    since this is its only user."""

    text: str
    is_final: bool
    confidence: float
    language_code: str
    # When this result ARRIVED (used to drop stale results, see stt_router.TurnGate)
    received_at: float = field(default_factory=time.monotonic)

# Flux model names. English-only uses flux-general-en (fastest, most accurate
# for English). Hindi requires the multilingual model with a language_hint --
# see the caveat about Hindi+English code-switch accuracy discussed earlier.
FLUX_MODEL_MAP = {
    "en": "flux-general-en",
    "en-hi": "flux-general-multi",
}
FLUX_LANGUAGE_HINTS = {"en-hi": ["en", "hi"]}

# Words Flux should specifically listen for - boosts recognition of
# names, hotels and cities that a general English model has no reason
# to expect, per https://developers.deepgram.com/reference/speech-to-text/listen-flux
#
# IMPORTANT: this used to be a single hardcoded list that included
# fictional/placeholder hotel names (e.g. "Sunvale Retreat"). Those
# didn't match what's actually in MongoDB (real hotels are things like
# "Vista Haven", "Serene Crown"), so Flux was never actually boosted
# for the words customers really say - it was boosted for words that
# never come up. That's exactly why "Vista Haven" got heard as "Wish
# to haven't" - the keyterm list and the live data had drifted apart.
#
# Fix: cities/hotels are now pulled LIVE from Mongo (see
# build_keyterms() below) each call, so this list can never go stale
# again. DEFAULT_KEYTERMS below is only a fallback for when the DB
# isn't reachable yet, or for anything not city/hotel data (email
# providers - these don't change and aren't in Mongo).
DEFAULT_KEYTERMS = [
    "gmail", "yahoo", "outlook", "hotmail", "icloud", "rediffmail",
]

# Kept as a max, matching Deepgram's practical limit for keyterm
# prompting - a huge list dilutes the boost, so if the DB ever has way
# more hotels than this, trim rather than sending everything.
_MAX_KEYTERMS = 100


def build_keyterms(cities: list[str] | None = None, hotels: list[str] | None = None) -> list[str]:
    """Combines live city/hotel names (fetched by the caller, e.g.
    main.py via mongo.get_cities()/mongo.get_all_hotel_names()) with
    the fixed email-provider list, so Flux is always boosted for the
    names actually in use today rather than a hardcoded snapshot."""
    terms = list(DEFAULT_KEYTERMS)
    for name in (cities or []) + (hotels or []):
        if name and name not in terms:
            terms.append(name)
    return terms[:_MAX_KEYTERMS]


def stream_transcribe(api_key: str, lang_key: str, audio_chunks, keyterms: list[str] | None = None):
    """
    api_key: your Deepgram API key
    lang_key: "en" (see FLUX_MODEL_MAP)
    audio_chunks: an iterable/generator of raw PCM16 bytes (16kHz, mono),
                  e.g. from audio_sources.py's MicrophoneStream or
                  FileAudioStream generators.
    keyterms: words to boost recognition for - pass the result of
              build_keyterms() with live city/hotel names. Falls back
              to DEFAULT_KEYTERMS (email providers only) if omitted.

    Yields TranscriptResult objects: interim updates first, then a final
    result (is_final=True) each time Flux detects the speaker's turn ended.
    """
    model = FLUX_MODEL_MAP.get(lang_key, "flux-general-en")
    client = DeepgramClient(api_key=api_key)

    connect_kwargs = dict(
        model=model,
        encoding="linear16",
        sample_rate=16000,
        eot_threshold=settings.FLUX_EOT_THRESHOLD,
        eot_timeout_ms=settings.FLUX_EOT_TIMEOUT_MS,
    )
    if model == "flux-general-en":
        # keyterm boosting is English-model only; don't send it to the multi model
        connect_kwargs["keyterm"] = keyterms if keyterms is not None else DEFAULT_KEYTERMS
    else:
        connect_kwargs["request_options"] = {
            "additional_query_parameters": {"language_hint": FLUX_LANGUAGE_HINTS[lang_key]}
        }
    results_queue: "queue.Queue" = queue.Queue()
    stop_event = threading.Event()

    with client.listen.v2.connect(**connect_kwargs) as connection:

        def on_message(message):
            # Flux's wire format nests the turn event inside a TurnInfo
            # message (message.type == "TurnInfo", message.event ==
            # "StartOfTurn"/"Update"/"EndOfTurn"/...). Some SDK versions
            # flatten this so message.type IS the event name directly.
            # Check both so this works either way.
            msg_type = getattr(message, "event", None) or getattr(message, "type", None)
            transcript = getattr(message, "transcript", "") or ""
            if not transcript:
                return

            is_final = msg_type == "EndOfTurn"

            # Flux reports confidence per word, not per message -- average
            # them as a reasonable stand-in for an overall confidence score.
            words = getattr(message, "words", None) or []
            if words:
                confidence = sum(w.confidence for w in words) / len(words)
            else:
                confidence = 0.0

            results_queue.put(
                TranscriptResult(
                    text=transcript,
                    is_final=is_final,
                    confidence=confidence,
                    language_code=detect_script_language(transcript, default="en"),
                )
            )

        def on_error(error):
            results_queue.put(RuntimeError(f"Deepgram error: {error}"))

        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)

        listener_thread = threading.Thread(
            target=connection.start_listening, daemon=True
        )
        listener_thread.start()

        def feed_audio():
            try:
                for chunk in audio_chunks:
                    if stop_event.is_set():
                        break
                    connection.send_media(chunk)
            except Exception as e:  # surface feeder errors through the same queue
                results_queue.put(e)

        feeder_thread = threading.Thread(target=feed_audio, daemon=True)
        feeder_thread.start()

        try:
            while feeder_thread.is_alive() or not results_queue.empty():
                try:
                    item = results_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop_event.set()