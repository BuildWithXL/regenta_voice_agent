"""
Local test harness: talk to the agent through your microphone or a WAV
file. Real phone calls go through plivo_server.py instead; both use the
same conversation loop (conversation.py).

Handles new bookings (booking_agent.py) and existing-booking support
(support_agent.py); ai_response.py routes between them.

Run from the voice-agent/ directory:
    python main.py --source mic --lang en
    python main.py --source file --file sample.wav --lang en
Say/type 'quit' or 'exit' to end the call early (recorded as
Stage.ABANDONED)."""

import asyncio
import argparse
import sys

import config
import mongo
import okf_store
from ai_response import opening_line, close as close_groq
from audio_sources import FileAudioStream, MicrophoneStream
from conversation import converse
from sarvam_stt_client import stream_transcribe
from session import Stage, store
from stt_router import TurnGate, with_turn_gate
from tts_client import close as close_tts, speak


async def _say(text: str, language: str = "en") -> None:
    """Prints and speaks a line. If TTS isn't available in this
    environment, don't let that kill the conversation loop itself."""
    print(f"AI: {text}")
    try:
        await speak(text, language)
    except Exception as exc:
        print(f"  (TTS playback skipped: {exc})")


async def _finish_call(session) -> None:
    """Report a transfer and save the call log."""
    if session.stage == Stage.TRANSFERRED:
        # On a real call plivo_server.py transfers the caller to HUMAN_AGENT_NUMBER.
        print(f"[handoff] {session.handoff_summary}")
    try:
        await mongo.save_call_log(session)
    except Exception as exc:  # a logging failure must never break the call
        print(f"[call log] not saved: {exc!r}")


async def run_call(args) -> None:
    session = await store.create()
    # Real calls take this from Plivo's "From"; locally use --caller-number to test.
    session.caller_number = args.caller_number

    stream = MicrophoneStream() if args.source == "mic" else FileAudioStream(args.file)
    gate = TurnGate() if args.source == "mic" else None   # a file can't pause for the AI

    with stream as audio_stream:
        mic_stream = audio_stream if args.source == "mic" else None
        if mic_stream is not None:
            mic_stream.muted = True
        await _say(opening_line(session), "en")
        if mic_stream is not None:
            mic_stream.muted = False
        raw_results = stream_transcribe(config.get_sarvam_api_key(), args.lang, audio_stream.generator())
        await converse(session, with_turn_gate(raw_results, gate), _say, mic_stream, gate)

    if session.stage == Stage.COMPLETED:
        print("\n[call ended — query resolved]")
    elif session.stage == Stage.ABANDONED:
        print("\n[call ended — customer hung up]")
    elif session.stage == Stage.TRANSFERRED:
        print(f"\n[call ended — transferred to human agent ({session.escalation_reason})]")
    await _finish_call(session)

async def main() -> None:
    args = parse_args()
    kb_task = None
    try:
        # Hotel knowledge base (hotel_kb/ locally, or S3 in production - chosen
        # by KB_STORAGE). A failure here must never stop the booking flow.
        try:
            await okf_store.start()
            kb_task = asyncio.create_task(okf_store.refresh_loop())
        except Exception as exc:
            print(f"  (knowledge base unavailable, continuing without it: {exc})")
        await run_call(args)
    finally:
        if kb_task is not None:
            kb_task.cancel()
            try:
                await kb_task
            except asyncio.CancelledError:
                pass
        await close_groq()
        await close_tts()
        await mongo.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Voice agent — local mic/file test")
    parser.add_argument(
        "--source",
        choices=["mic", "file"],
        required=True,
        help="Where to get audio from: your microphone, or a WAV file",
    )
    parser.add_argument(
        "--file",
        help="Path to a 16-bit PCM WAV file (required when --source file)",
    )
    parser.add_argument(
        "--lang",
        choices=config.SUPPORTED_LANGUAGES,
        required=True,
        help="Language to expect: 'en' (English) or 'hi' (Hindi)",
    )
    parser.add_argument("--caller-number", default=None,
                        help="Caller's phone number (testing only; Plivo calls set it)")
    args = parser.parse_args()

    if args.source == "file" and not args.file:
        parser.error("--file is required when --source file")

    return args


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
