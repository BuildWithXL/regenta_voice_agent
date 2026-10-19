"""
Phase 9 — AI Response -> TTS (Sarvam Bulbul v3).
Phase 10 — Audio Playback.

Uses Sarvam's HTTP streaming endpoint (POST /text-to-speech/stream):
the response is a raw binary audio stream, so playback starts as soon
as the first audio arrives instead of after the whole utterance is
synthesised. We request linear16 (raw 16-bit PCM) at 24 kHz - the
highest rate the streaming endpoint supports - which is exactly what
the local sounddevice output stream below plays, so no decoding step.

Bulbul v3 speaks English (Indian accent), Hindi, Gujarati and 8 other
Indian languages. The language code is picked from the SCRIPT of the
text being spoken (Devanagari -> hi-IN, Gujarati -> gu-IN, otherwise
en-IN), so replies in the caller's language are voiced correctly.

Phone calls (plivo_server.py) use synthesize() instead: same request at
the call's sample rate, PCM chunks handed back instead of played locally.
"""
from typing import AsyncIterator

import httpx

from config import LANGUAGES, settings, require
from language_utils import detect_script_language

_SAMPLE_RATE = 24000          # speech_sample_rate requested below
_BYTES_PER_FRAME = 2          # int16, mono
# Hold back ~150 ms of audio before starting playback, so a slow network
# chunk can't make the speakers under-run (clicks/gaps) mid-sentence.
_PREBUFFER_BYTES = int(_SAMPLE_RATE * _BYTES_PER_FRAME * 0.15)
_MAX_CHARS = 3500             # Sarvam's per-request limit
_LANG_CODES = {"en": "en-IN", "hi": "hi-IN", "gu": "gu-IN"}
_LANG_CODES.update({k: v["tts_code"] for k, v in LANGUAGES["languages"].items() if v.get("tts_code")})
# TTS code -> voice for that language (languages.json "voice"; empty = SARVAM_TTS_SPEAKER)
_VOICES = {v["tts_code"]: v["voice"] for v in LANGUAGES["languages"].values()
           if v.get("tts_code") and v.get("voice")}

_client: httpx.AsyncClient | None = None


class _FrameAligner:
    """HTTP chunks can end mid-sample (odd byte count). Only ever hand
    out whole 2-byte frames; carry the leftover byte to the next chunk."""

    def __init__(self) -> None:
        self._leftover = b""

    def feed(self, chunk: bytes) -> bytes:
        data = self._leftover + chunk
        usable_len = len(data) - (len(data) % _BYTES_PER_FRAME)
        self._leftover = data[usable_len:]
        return data[:usable_len]


def _strip_wav_header(buf: bytes) -> bytes:
    """If the stream starts with a WAV header, drop it - otherwise the
    header bytes would be played as a click of noise."""
    if buf[:4] == b"RIFF":
        i = buf.find(b"data")
        if i != -1 and len(buf) >= i + 8:
            return buf[i + 8:]
    return buf


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            headers={
                "api-subscription-key": require("SARVAM_API_KEY"),
                "Content-Type": "application/json",
            },
            timeout=15.0,
            # Calls are seconds apart (STT + LLM in between); don't keep
            # idle connections that the server may have already closed.
            limits=httpx.Limits(max_keepalive_connections=0),
        )
    return _client


def _resolve_language(text: str, language_code: str | None) -> str:
    if language_code in _LANG_CODES:
        return _LANG_CODES[language_code]
    if language_code in _LANG_CODES.values():                  # already "hi-IN"
        return language_code
    # None or not a TTS language ("auto", "en-hi"): detect from the text
    return _LANG_CODES[detect_script_language(text, default="en")]


async def synthesize(text: str, language_code: str | None = None,
                     sample_rate: int = _SAMPLE_RATE) -> AsyncIterator[bytes]:
    """Yields raw PCM16 mono audio for `text` as it streams in from Sarvam
    (WAV header stripped, whole 2-byte frames only). language_code is
    optional ("en"/"hi"/"gu" or "hi-IN"); by default it is detected from
    the text."""
    global _client
    text = (text or "").strip()
    if not text:
        return

    target = _resolve_language(text, language_code)
    payload = {
        "text": text[:_MAX_CHARS],
        "language_code": target,
        "speaker": _VOICES.get(target, settings.SARVAM_TTS_SPEAKER),
        "model": settings.SARVAM_TTS_MODEL,
        "output_audio_codec": "linear16",
        "speech_sample_rate": sample_rate,
        "pace": settings.SARVAM_TTS_PACE,
        "temperature": settings.SARVAM_TTS_TEMPERATURE,
        "enable_preprocessing": True,     # numbers, prices, mixed-language text
    }

    for attempt in range(2):
        aligner = _FrameAligner()
        head = b""            # first bytes, held until the WAV header is known
        yielded = False
        try:
            async with _get_client().stream("POST", settings.SARVAM_TTS_URL, json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")[:300]
                    raise RuntimeError(f"Sarvam TTS error {response.status_code}: {body}")
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    if head is not None:
                        head += chunk
                        if len(head) < 64:
                            continue
                        chunk, head = _strip_wav_header(head), None
                    aligned = aligner.feed(chunk)
                    if aligned:
                        yielded = True
                        yield aligned
            if head:                          # tiny reply, under 64 bytes
                aligned = aligner.feed(_strip_wav_header(head))
                if aligned:
                    yield aligned
            return
        except (httpx.RemoteProtocolError, httpx.ReadError):
            # Stale/closed connection. Retry once with a fresh client - but
            # only if nothing was sent yet, or the caller would hear it twice.
            if attempt or yielded:
                raise
            _client = None


async def speak(text: str, language_code: str | None = None) -> None:
    """Speak `text` through the local speakers (mic/file mode)."""
    import sounddevice as sd      # only needed locally; servers have no audio device

    out = sd.RawOutputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16")
    started = False
    pending = b""
    try:
        async for data in synthesize(text, language_code, _SAMPLE_RATE):
            if not started:
                # Hold back ~150 ms so a slow network chunk can't under-run the speakers.
                pending += data
                if len(pending) < _PREBUFFER_BYTES:
                    continue
                data, pending = pending, b""
                out.start()
                started = True
            out.write(data)
        if pending:                          # short reply, under the prebuffer
            out.start()
            started = True
            out.write(pending)
    finally:
        if started:
            out.stop()          # waits for the buffered audio to finish playing
        out.close()


async def interrupt() -> None:
    """Placeholder for barge-in handling - belongs in the Voice Gateway
    layer once this moves onto real telephony, not this local prototype."""
    raise NotImplementedError(
        "Barge-in interruption belongs in the Voice Gateway layer, not "
        "this local prototype's TTS client."
    )


async def close() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None