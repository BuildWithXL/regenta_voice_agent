import asyncio, base64, queue, threading

from sarvamai import AsyncSarvamAI, RealtimeAudioInput, RealtimeEnd

from config import settings
from language_utils import detect_script_language
from stt_client import TranscriptResult

_LANG_MAP = {"en-IN": "en", "hi-IN": "hi", "gu-IN": "gu"}
_SILENCE_100MS = b"\x00" * 3200      # 100ms of 16kHz mono PCM16


def stream_transcribe(api_key, lang_key, audio_chunks, keyterms=None):
    results: "queue.Queue" = queue.Queue()
    stop = threading.Event()

    async def run():
        client = AsyncSarvamAI(api_subscription_key=api_key)
        loop = asyncio.get_running_loop()
        audio_q: asyncio.Queue = asyncio.Queue()

        def pump():  # plain thread: drains your sync audio generator
            try:
                for chunk in audio_chunks:
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(audio_q.put_nowait, chunk)
            except Exception as e:
                results.put(e)
            finally:
                loop.call_soon_threadsafe(audio_q.put_nowait, None)

        threading.Thread(target=pump, daemon=True).start()

        async with client.speech_to_text_realtime_streaming.connect(
            language_code="auto",
            model=settings.SARVAM_STT_MODEL,
            mode=settings.SARVAM_STT_OUTPUT_MODE,
            stream_type="fast",
            silence_duration_ms=settings.SARVAM_SILENCE_MS,
        ) as ws:

            async def send():
                while True:
                    try:
                        chunk = await asyncio.wait_for(audio_q.get(), timeout=10)
                    except asyncio.TimeoutError:
                        # mic is muted while the AI speaks -> no audio flows.
                        # Send silence so the socket isn't closed for inactivity.
                        if stop.is_set():
                            break
                        chunk = _SILENCE_100MS
                    if chunk is None:
                        break
                    await ws.send_realtime_audio_input(
                        RealtimeAudioInput(audio=base64.b64encode(chunk).decode("utf-8"))
                    )
                await ws.send_realtime_end(RealtimeEnd())

            async def receive():
                async for m in ws:
                    ev = getattr(m, "event", None)
                    if ev in ("transcript.partial", "transcript.final"):
                        text = (getattr(m, "text", "") or "").strip()
                        if not text:
                            continue
                        lang = _LANG_MAP.get(getattr(m, "language", None)) \
                            or detect_script_language(text)
                        results.put(TranscriptResult(
                            text=text,
                            is_final=(ev == "transcript.final"),
                            confidence=1.0,   # Sarvam gives no per-word confidence
                            language_code=lang,
                        ))
                    elif ev == "error":
                        results.put(RuntimeError(f"Sarvam error {m.code}: {m.message}"))
                        if getattr(m, "is_fatal", False):
                            return

            await asyncio.gather(send(), receive())

    def worker_main():
        try:
            asyncio.run(run())
        except Exception as e:
            results.put(e)

    worker = threading.Thread(target=worker_main, daemon=True)
    worker.start()
    try:
        while worker.is_alive() or not results.empty():
            try:
                item = results.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()