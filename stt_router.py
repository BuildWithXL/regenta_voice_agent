import queue
import threading
import time

import config
import stt_sarvam
from stt_client import TranscriptResult
from stt_client import stream_transcribe as flux_stream


class TurnGate:
    """main.py calls ai_finished() when the AI has finished speaking. Every
    STT result that ARRIVED before that moment (+ a short tail) was spoken
    while the AI was thinking/talking and is dropped."""

    def __init__(self) -> None:
        self.open_after = 0.0

    def ai_finished(self, tail_s: float = 0.4) -> None:
        self.open_after = time.monotonic() + tail_s

    def accepts(self, received_at: float) -> bool:
        return received_at >= self.open_after


def _with_turn_grace(source, gate, grace_s):
    """1. drops stale results (TurnGate);
       2. holds each FINAL for grace_s - if the caller keeps talking, the
          parts are merged into one turn (natural pauses don't cut them off)."""
    q: "queue.Queue" = queue.Queue()
    done = object()
    stopped = threading.Event()

    def pump():
        try:
            for item in source:
                if stopped.is_set():
                    break
                q.put(item)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(done)

    threading.Thread(target=pump, daemon=True).start()

    held = None
    deadline = 0.0
    try:
        while True:
            wait = 0.5 if held is None else min(max(deadline - time.monotonic(), 0.0), 0.5)
            try:
                item = q.get(timeout=wait)
            except queue.Empty:
                if held is not None and time.monotonic() >= deadline:
                    out, held = held, None
                    yield out
                continue

            if item is done:
                if held is not None:
                    yield held
                return
            if isinstance(item, Exception):
                if held is not None:
                    yield held
                raise item
            if gate is not None and not gate.accepts(item.received_at):
                continue                      # stale: spoken during the AI's turn

            if not item.is_final:
                if held is not None:
                    deadline = time.monotonic() + grace_s   # caller still talking
                yield item
                continue

            if held is None:
                held = item
            else:                              # caller resumed: merge into one turn
                held = TranscriptResult(
                    text=f"{held.text} {item.text}".strip(),
                    is_final=True,
                    confidence=min(held.confidence, item.confidence),
                    language_code=item.language_code,
                    received_at=item.received_at,
                )
            deadline = time.monotonic() + grace_s
    finally:
        stopped.set()


def _raw_stream(api_key, lang_key, audio_chunks, keyterms=None):
    if lang_key == "auto":
        yielded = False
        try:
            for r in stt_sarvam.stream_transcribe(
                config.get_sarvam_api_key(), lang_key, audio_chunks, keyterms
            ):
                yielded = True
                yield r
            return
        except Exception as exc:
            if yielded:
                raise
            print(f"  (Sarvam unavailable, falling back to Flux English: {exc})")
            lang_key = "en"
    yield from flux_stream(api_key, lang_key, audio_chunks, keyterms=keyterms)


def stream_transcribe(api_key, lang_key, audio_chunks, keyterms=None, gate=None):
    raw = _raw_stream(api_key, lang_key, audio_chunks, keyterms)
    return with_turn_gate(raw, gate)


def with_turn_gate(source, gate=None):
    """Applies TurnGate + turn-grace merging to any STT result stream
    (e.g. sarvam_stt_client.stream_transcribe)."""
    return _with_turn_grace(source, gate, config.settings.TURN_GRACE_SECONDS)