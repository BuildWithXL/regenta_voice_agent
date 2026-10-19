"""Central configuration for the support voice agent."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def require(name: str) -> str:
    """Raise clearly if a required env var is missing. Called lazily from the
    module that needs it, so importing config on its own is always safe."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


class Settings:

    # --- Backend Reply -> AI Response ---
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    # --- Phase 9: TTS (Deepgram Aura-2) ---
    # One Deepgram key covers both STT (Flux, not built yet) and TTS
    # (Aura-2, built now) — per the doc's note that no separate
    # cloud-provider-style setup is needed.
    
    # Thalia is a warm, friendly voice — a reasonable starting pick for
    # a hotel receptionist per the doc's Phase 9 requirement. Try a few
    # alternatives (aura-2-luna-en, aura-2-stella-en) and compare.

    # --- TTS: Sarvam Bulbul v3 ---
    SARVAM_TTS_URL = os.getenv("SARVAM_TTS_URL", "https://api.sarvam.ai/text-to-speech/stream")
    SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
    SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "shreya")   # any Bulbul v3 voice
    SARVAM_TTS_PACE = float(os.getenv("SARVAM_TTS_PACE", "1.0"))          # 0.5 - 2.0
    SARVAM_TTS_TEMPERATURE = float(os.getenv("SARVAM_TTS_TEMPERATURE", "0.6"))  # 0.01 - 1.0

    SAMPLE_RATE = 16000  # used later once STT/mic capture is built
    CHANNELS = 1

    # --- STT routing (Sarvam, used when --lang auto) ---
    SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3-realtime")
    # "codemix" | "transcribe" | "translit" -- compare these three on real calls
    SARVAM_STT_OUTPUT_MODE = os.getenv("SARVAM_STT_OUTPUT_MODE", "codemix")
    # Silence before a turn ends. Higher = doesn't cut off callers who pause
    # while spelling an email. Flux equivalent was eot_timeout_ms.
    SARVAM_SILENCE_MS = int(os.getenv("SARVAM_SILENCE_MS", "1800"))

    # --- Turn-taking (how patient the agent is) ---
    FLUX_EOT_THRESHOLD = float(os.getenv("FLUX_EOT_THRESHOLD", "0.9"))
    FLUX_EOT_TIMEOUT_MS = int(os.getenv("FLUX_EOT_TIMEOUT_MS", "8000"))
    # Extra wait after a turn ends; if the caller carries on inside it,
    # both parts are merged into ONE answer.
    TURN_GRACE_SECONDS = float(os.getenv("TURN_GRACE_SECONDS", "1.2"))
    # Ignore STT results for this long after the AI finishes speaking.
    POST_TTS_TAIL_SECONDS = float(os.getenv("POST_TTS_TAIL_SECONDS", "0.4"))

    # --- Phase 7: Session storage / MongoDB ---
    # Only required once a booking actually needs to be persisted
    # (session/mongo.py calls require() lazily, same pattern as TTS).
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "voice_agent")
    MONGODB_HOTELS_COLLECTION = os.getenv("MONGODB_HOTELS_COLLECTION", "hotels")
    MONGODB_BOOKINGS_COLLECTION = os.getenv(
        "MONGODB_BOOKINGS_COLLECTION", "bookings"
    )


    # In-memory session cleanup (call SessionStore.cleanup_idle() on a timer)
    SESSION_IDLE_TIMEOUT_SECONDS = int(os.getenv("SESSION_IDLE_TIMEOUT_SECONDS", "1800"))

    # STT results below this confidence are ignored (short yes/no answers are exempt)
    STT_MIN_CONFIDENCE = float(os.getenv("STT_MIN_CONFIDENCE", "0.85"))

    # After this many failed booking lookups in one call, hand over to a human
    MAX_FAILED_VERIFICATIONS = int(os.getenv("MAX_FAILED_VERIFICATIONS", "3"))

    # Guardrails
    MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "500"))          # longer caller turns are cut
    MAX_INJECTION_ATTEMPTS = int(os.getenv("MAX_INJECTION_ATTEMPTS", "3"))  # then a human takes over
    MAX_TOOL_CALLS_PER_CALL = int(os.getenv("MAX_TOOL_CALLS_PER_CALL", "20"))

    # Hotel FAQ data used by the getHotelInfo tool
    HOTEL_INFO_PATH = os.getenv("HOTEL_INFO_PATH", "hotel_info.json")

    # --- Phone calls: Plivo (plivo_server.py) ---
    PLIVO_NUMBER = os.getenv("PLIVO_NUMBER", "")    # the number callers dial; transfers' caller ID
    # Public https URL of this server, e.g. https://voice.example.com (no trailing slash).
    PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    HUMAN_AGENT_NUMBER = os.getenv("HUMAN_AGENT_NUMBER", "")      # transfers dial this
    PHONE_STT_LANG = os.getenv("PHONE_STT_LANG", "auto")          # "en" | "en-hi" | "auto"
    PHONE_SAMPLE_RATE = 16000     # Plivo stream audio/x-l16;rate=16000, both directions
    # Turn off only for local tests without real Plivo requests.
    PLIVO_VALIDATE_SIGNATURE = os.getenv("PLIVO_VALIDATE_SIGNATURE", "true").lower() != "false"


settings = Settings()


def _load_languages() -> dict:
    """Call languages (languages.json): per language its name, STT code/mode,
    TTS code, voice, aliases and fixed lines. Adding a language = a new entry."""
    import json
    path = os.getenv("LANGUAGES_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "languages.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


LANGUAGES = _load_languages()

# Languages selectable via the CLI's --lang flag.
SUPPORTED_LANGUAGES = ["en", "en-hi", "auto"]

# Audio format assumptions used throughout the project.
SAMPLE_RATE_HZ = 16000
CHANNELS = 1
CHUNK_SIZE_FRAMES = 1600  # ~100 ms of audio per chunk at 16 kHz


def get_sarvam_api_key() -> str:
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        sys.exit("Missing SARVAM_API_KEY. Get one at https://dashboard.sarvam.ai")
    return api_key
