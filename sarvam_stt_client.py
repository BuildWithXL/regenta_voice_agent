"""
Sarvam realtime streaming STT (saaras:v4 / saaras:v3-realtime).

Drop-in replacement for stt_client.stream_transcribe(): same signature,
same TranscriptResult shape, still a plain synchronous generator, so
main.py's loop doesn't change. Internally it runs Sarvam's async
WebSocket client on a background thread and hands results back through
a queue.

Note: Sarvam's realtime transcripts carry no per-utterance confidence,
so `confidence` is always 1.0 and main.py's MIN_CONFIDENCE check never
fires with this backend.
"""
import asyncio
import base64
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field

from sarvamai import AsyncSarvamAI, RealtimeAudioInput, RealtimeEnd, RealtimePing

STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")  # or "saaras:v3-realtime"
SILENCE_MS = os.getenv("SARVAM_STT_SILENCE_MS", "1000")

LANGUAGE_CODES = {"en": "en-IN", "hi": "hi-IN", "auto": "auto"}

# saaras:v4 only: biases recognition toward these (max 50, 64 chars each).
# Put hotel-specific words here: your city list, room types, brand name.
KEYTERMS = []  # add only city names later, after name capture works

# Terminology hint applied to final transcripts.
PROMPT = ""  # test without a prompt first

PING_EVERY_S = 3.0  # mic is muted while TTS plays; keep the socket alive
MAX_FRAME_BYTES = 8000  # must be even (int16 audio) and under Sarvam's 16000-byte cap

@dataclass
class TranscriptResult:
    text: str
    is_final: bool
    confidence: float
    language_code: str
    received_at: float = field(default_factory=time.monotonic)  # for stt_router.TurnGate


def _short_lang(detected: str | None, lang_key: str) -> str:
    """'hi-IN' -> 'hi'. Never returns 'auto'/'en-hi': main.py stores this
    as session.language and TTS needs a real language."""
    code = (detected or "").split("-")[0].lower()
    if code in ("en", "hi", "gu"):
        return code
    return lang_key if lang_key in ("en", "hi", "gu") else "en"


def stream_transcribe(api_key: str, lang_key: str, audio_chunks):
    """Yields TranscriptResult: partials first, then one final per turn."""
    results: "queue.Queue" = queue.Queue()
    stop = threading.Event()
    done = object()

    async def _run():
        loop = asyncio.get_running_loop()
        client = AsyncSarvamAI(api_subscription_key=api_key)

        kwargs = dict(
            language_code=LANGUAGE_CODES.get(lang_key, "en-IN"),
            model=STT_MODEL,
            stream_type="fast",
        
            silence_duration_ms=SILENCE_MS,
        )
        if PROMPT:
            kwargs["prompt"] = PROMPT
        if STT_MODEL == "saaras:v4" and KEYTERMS:
            kwargs["keyterms"] = json.dumps(KEYTERMS)

        async with client.speech_to_text_realtime_streaming.connect(**kwargs) as ws:

            async def send_audio():
                it = iter(audio_chunks)
                while not stop.is_set():
                    # next() blocks (mic queue), so run it off-loop
                    chunk = await loop.run_in_executor(None, next, it, None)
                    if chunk is None:
                        break
                    for i in range(0, len(chunk), MAX_FRAME_BYTES):
                        piece = chunk[i:i + MAX_FRAME_BYTES]
                        await ws.send_realtime_audio_input(
                            RealtimeAudioInput(audio=base64.b64encode(piece).decode())
                        )
                await ws.send_realtime_end(RealtimeEnd())

            async def keepalive():
                while not stop.is_set():
                    await asyncio.sleep(PING_EVERY_S)
                    await ws.send_realtime_ping(RealtimePing())

            async def receive():
                async for m in ws:
                    ev = getattr(m, "event", None)
                    if ev in ("transcript.partial", "transcript.final"):
                        text = (getattr(m, "text", "") or "").strip()
                        if text:
                            results.put(TranscriptResult(
                                text=text,
                                is_final=(ev == "transcript.final"),
                                confidence=1.0,
                                language_code=_short_lang(
                                    getattr(m, "language_code", None), lang_key
                                ),
                            ))
                    elif ev == "error":
                        results.put(RuntimeError(
                            f"Sarvam STT error ({m.code}): {m.message}"
                        ))
                        if getattr(m, "is_fatal", False):
                            return

            ka = asyncio.create_task(keepalive())
            try:
                await asyncio.gather(send_audio(), receive())
            finally:
                ka.cancel()

    def _thread_main():
        try:
            asyncio.run(_run())
        except Exception as exc:  # surface through the queue
            results.put(exc)
        finally:
            results.put(done)

    t = threading.Thread(target=_thread_main, daemon=True)
    t.start()
    try:
        while True:
            item = results.get()
            if item is done:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()