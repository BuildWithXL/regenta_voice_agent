"""
Sarvam Bulbul v3 TTS over HTTP streaming (POST /text-to-speech/stream).

Drop-in replacement for tts_client.py: speak(text, language_code) and
close(), with the same chunk-by-chunk playback and frame alignment.

The stream endpoint returns raw binary audio (not base64 JSON).
HTTP streaming is capped at 24 kHz, which is what we use.

SARVAM_TTS_CODEC defaults to "linear16" (raw PCM16) so it can go
straight into sounddevice. If the API rejects that value, check the
current /text-to-speech/stream reference for the exact PCM codec name.
"""
import os

import httpx
import sounddevice as sd

from config import require

_URL = "https://api.sarvam.ai/text-to-speech/stream"
_SAMPLE_RATE = 24000
_BYTES_PER_FRAME = 2  # int16 mono

SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "priya")
MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
CODEC = os.getenv("SARVAM_TTS_CODEC", "linear16")
PACE = float(os.getenv("SARVAM_TTS_PACE", "1.0"))

_LANG = {"en": "en-IN", "hi": "hi-IN", "gu": "gu-IN"}

_client: httpx.AsyncClient | None = None


class _FrameAligner:
    def __init__(self) -> None:
        self._leftover = b""

    def feed(self, chunk: bytes) -> bytes:
        data = self._leftover + chunk
        usable = len(data) - (len(data) % _BYTES_PER_FRAME)
        self._leftover = data[usable:]
        return data[:usable]


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            headers={"api-subscription-key": require("SARVAM_API_KEY")},
            timeout=20.0,
        )
    return _client


async def speak(text: str, language_code: str | None = None) -> None:
    if not text.strip():
        return

    payload = {
        "text": text[:3500],  # stream endpoint limit
        "target_language_code": _LANG.get(language_code or "en", "en-IN"),
        "speaker": SPEAKER,
        "model": MODEL,
        "pace": PACE,
        "speech_sample_rate": _SAMPLE_RATE,
        "output_audio_codec": CODEC,
    }

    out = sd.RawOutputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16")
    out.start()
    aligner = _FrameAligner()
    first = True
    try:
        async with _get_client().stream("POST", _URL, json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                if first:
                    first = False
                    if chunk[:4] == b"RIFF":  # skip a WAV header if present
                        chunk = chunk[44:]
                aligned = aligner.feed(chunk)
                if aligned:
                    out.write(aligned)
    finally:
        out.stop()
        out.close()


async def close() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None