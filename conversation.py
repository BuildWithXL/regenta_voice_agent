"""
One call's conversation loop, shared by the local CLI (main.py: mic/file)
and real phone calls (plivo_server.py).

converse() reads STT results, handles the "transfer you to a human?" answer,
cleans up the caller's words, asks ai_response for the reply and speaks it
through the `say` callback it was given. It returns when the call reaches an
end stage or the audio stops.
"""
import asyncio
import time
from typing import Awaitable, Callable, Iterable

import config
import call_language
from ai_response import build_handoff_summary, generate_reply
from language_utils import (
    collapse_spelled_letters, normalize_spoken_booking_id, normalize_spoken_symbols,
)
from session import Session, Stage
from stt_router import TurnGate

Say = Callable[[str, str], Awaitable[None]]   # say(text, language)

# The "transferring" / "continue" lines are per language in languages.json.

_YES = {"yes", "yeah", "yep", "sure", "please", "ok", "okay",
        "haan", "han", "haanji", "हाँ", "हां", "जी", "ठीक", "जरूर", "ज़रूर"}
_NO = {"no", "nope", "nahi", "nahin", "नहीं", "नही"}

# Stages where there's nothing left for the customer to say.
END_STAGES = {Stage.COMPLETED, Stage.TRANSFERRED, Stage.ABANDONED}
MIN_CONFIDENCE = config.settings.STT_MIN_CONFIDENCE  # below this, speech is too unreliable to act on


def _words(text: str) -> set:
    return set(text.lower().replace(",", " ").replace(".", " ").replace("।", " ").split())


def _is_yes(text: str) -> bool:
    w = _words(text)
    return bool(w & _YES) and not (w & _NO)


def _is_no(text: str) -> bool:
    return bool(_words(text) & _NO)


async def converse(session: Session, results: Iterable, say: Say, audio=None,
                   gate: TurnGate | None = None, tag: str = "") -> None:
    """results: STT results (a blocking generator; read on a worker thread so
    a phone server's event loop keeps receiving audio meanwhile).
    audio: the audio source, muted while the AI thinks and speaks (or None).
    gate: drops speech that arrived during the AI's turn (or None)."""
    it = iter(results)
    while session.stage not in END_STAGES:
        result = await asyncio.to_thread(next, it, None)
        if result is None:
            return
        if not result.is_final:
            if not tag:
                print(f"\r[interim] {result.text}", end="", flush=True)
            continue

        print(f"\r{tag}You: {result.text}   (confidence: {result.confidence:.2f})")

        # Answer to "transfer you to a human?" (support flow) - checked
        # BEFORE anything else because short answers like "yes" are enough.
        if session.awaiting_transfer:
            lang = session.language
            if _is_yes(result.text):
                session.escalation_reason = "customer_accepted_transfer"
                session.handoff_summary = build_handoff_summary(session, "customer_accepted_transfer")
                session.set_stage(Stage.TRANSFERRED)
                await say(await call_language.line(session, "transferring"), lang)
                return
            session.awaiting_transfer = False
            if _is_no(result.text):
                await say(await call_language.line(session, "continue"), lang)
                continue
            # unclear answer: treat it as a normal turn

        raw_text = normalize_spoken_booking_id(normalize_spoken_symbols(
            collapse_spelled_letters(result.text.strip())
        ))
        if not raw_text:
            continue
        if raw_text.lower().strip(" .!") in {"quit", "exit"}:
            session.set_stage(Stage.ABANDONED)
            return

        user_text = raw_text
        # Once the caller has chosen a language it stays (Hinglish words don't switch it).
        session.language = session.call_language or result.language_code
        if result.language_code != "en" and not session.call_language:
            names = {"hi": "Hindi", "gu": "Gujarati"}
            user_text = f"[caller spoke {names.get(result.language_code, result.language_code)}] {user_text}"

        if result.confidence < MIN_CONFIDENCE:
            user_text = f"[transcription uncertain, confidence {result.confidence:.2f}] {user_text}"

        # The AI's turn: mute from NOW (thinking + speaking); afterwards
        # tell the STT layer to ignore anything that arrived meanwhile.
        if audio is not None:
            audio.muted = True
        try:
            t0 = time.perf_counter()
            reply = await generate_reply(session, user_text)
            print(f"  {tag}[timing] AI thinking: {time.perf_counter() - t0:.2f}s")
            await say(reply, session.language)
        finally:
            if audio is not None:
                audio.muted = False
            if gate is not None:
                gate.ai_finished(config.settings.POST_TTS_TAIL_SECONDS)
