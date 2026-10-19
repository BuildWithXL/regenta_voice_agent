"""
Phone calls through Plivo (number: settings.PLIVO_NUMBER).

    Caller -> Plivo number -> POST /plivo/answer
           <- XML: stream this call (both ways) to wss://<host>/plivo/stream
    Plivo <-> WS /plivo/stream: caller audio in (PCM16 16 kHz)
              -> Sarvam STT -> ai_response -> Sarvam TTS -> playAudio out

Plivo console -> Voice -> Applications -> your app:
    Answer URL : {PUBLIC_BASE_URL}/plivo/answer   (POST)
    Hangup URL : {PUBLIC_BASE_URL}/plivo/hangup   (POST)
then link the number to that application.

Run:
    uvicorn plivo_server:app --host 0.0.0.0 --port 8000

Sessions live in memory (session.store): run ONE instance of this server.
"""
import asyncio
import base64
import json
from contextlib import asynccontextmanager

import plivo
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from plivo import plivoxml
from plivo.utils.signature_v3 import validate_v3_signature

import config
import mongo
import okf_store
from ai_response import build_handoff_summary, close as close_groq, opening_line
from audio_sources import PhoneAudioStream
from config import require, settings
from conversation import END_STAGES, _TRANSFERRING, converse
from regenta_flow import dial_number
from sarvam_stt_client import stream_transcribe
from session import Session, Stage, store
from stt_router import TurnGate, with_turn_gate
from tts_client import close as close_tts, synthesize

_RATE = settings.PHONE_SAMPLE_RATE
_FRAME_BYTES = _RATE * 2 // 10         # 100 ms of PCM16 per playAudio message

# CallUUID -> caller number, from /plivo/answer until the stream starts.
# A stream whose callId isn't here was not announced by Plivo and is refused.
_pending_calls: dict[str, str] = {}
# CallUUID -> {"number", "what"} its transfer dials (the hotel's team, or HUMAN_AGENT_NUMBER).
_transfer_targets: dict[str, dict[str, str]] = {}

_DIAL_TIMEOUT = 25                   # seconds the human's phone rings before the fallback
_UNAVAILABLE = ("Sorry, our team is unavailable right now. "
                "Please call us again in a little while. Goodbye.")
_TRANSFER_FAILED = ("Sorry, I couldn't connect your call right now. "
                    "Please call us again in a little while. Goodbye.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    kb_task = None
    try:
        await okf_store.start()
        kb_task = asyncio.create_task(okf_store.refresh_loop())
    except Exception as exc:          # the KB must never stop calls
        print(f"  (knowledge base unavailable, continuing without it: {exc})")
    try:
        yield
    finally:
        if kb_task is not None:
            kb_task.cancel()
        await close_groq()
        await close_tts()
        await mongo.close()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Plivo webhooks
# ---------------------------------------------------------------------------
def _public_url(path: str) -> str:
    return f"{settings.PUBLIC_BASE_URL or require('PUBLIC_BASE_URL')}{path}"


async def _plivo_params(request: Request) -> dict[str, str]:
    """The webhook's form fields, after checking it really came from Plivo."""
    params = {k: v for k, v in (await request.form()).items()}
    if settings.PLIVO_VALIDATE_SIGNATURE:
        signature = request.headers.get("X-Plivo-Signature-V3", "")
        nonce = request.headers.get("X-Plivo-Signature-V3-Nonce", "")
        # Plivo signs the public URL it called, not the one behind the proxy.
        url = _public_url(request.url.path)
        if request.url.query:
            url += "?" + request.url.query
        if not (signature and nonce and validate_v3_signature(
                "POST", url, nonce, require("PLIVO_AUTH_TOKEN"), signature, dict(params))):
            raise HTTPException(status_code=403, detail="invalid Plivo signature")
    return params


def _xml(element) -> Response:
    return Response(element.to_string(), media_type="application/xml")


def _plivo_number(number: str) -> str:
    """Plivo wants numbers as digits with country code: '+91 22…' -> '9122…'."""
    return "".join(ch for ch in number if ch.isdigit())


def _mask(number: str | None) -> str:
    """'919902954007' -> '9199******07' for logs."""
    d = _plivo_number(number or "")
    return d[:4] + "*" * max(0, len(d) - 6) + d[-2:] if len(d) > 6 else (d or "-")


@app.post("/plivo/answer")
async def answer(request: Request) -> Response:
    params = await _plivo_params(request)
    call_uuid = params.get("CallUUID", "")
    _pending_calls[call_uuid] = params.get("From", "")
    print(f"[plivo] incoming call {call_uuid} from {params.get('From')} to {params.get('To')}")

    stream_url = _public_url("/plivo/stream").replace("https://", "wss://", 1)
    response = plivoxml.ResponseElement()
    response.add(plivoxml.StreamElement(
        stream_url,
        bidirectional=True,
        keepCallAlive=True,
        contentType=f"audio/x-l16;rate={_RATE}",
    ))
    return _xml(response)


@app.post("/plivo/transfer")
async def transfer(request: Request) -> Response:
    """XML Plivo fetches for the transferred A-leg (aleg_url in _transfer_call):
    <Dial callerId=PLIVO_NUMBER><Number>91XXXXXXXXXX</Number></Dial>, then the
    fallback message if nobody answers."""
    params = await _plivo_params(request)
    call_uuid = params.get("CallUUID", "")
    target = _transfer_targets.get(call_uuid) or {"number": settings.HUMAN_AGENT_NUMBER, "what": "human agent"}
    human, caller_id = dial_number(target["number"]), dial_number(settings.PLIVO_NUMBER)
    response = plivoxml.ResponseElement()
    if not (human and caller_id):
        print(f"[transfer] XML hit call={call_uuid} {target['what']}: no valid "
              f"{'number' if not human else 'PLIVO_NUMBER caller ID'} -> fallback message")
        response.add(plivoxml.SpeakElement(_TRANSFER_FAILED))
        response.add(plivoxml.HangupElement())
        return _xml(response)
    legs = {_plivo_number(params.get("From", ""))[-10:], _plivo_number(params.get("To", ""))[-10:]}
    print(f"[transfer] XML hit call={call_uuid} {target['what']} -> Dial {_mask(human)} "
          f"callerId={_mask(caller_id)} timeout={_DIAL_TIMEOUT}s")
    if human[-10:] in legs:
        print("[transfer] WARNING: the transfer number is the same phone as this call; it will ring busy")
    dial = plivoxml.DialElement(caller_id=caller_id, timeout=_DIAL_TIMEOUT,
                                action=_public_url("/plivo/transfer-status"), method="POST")
    dial.add(plivoxml.NumberElement(human))
    response.add(dial)
    return _xml(response)


@app.post("/plivo/transfer-status")
async def transfer_status(request: Request) -> Response:
    """Plivo posts here when the <Dial> ends. If the human never talked
    to the caller, tell the caller instead of dropping into silence."""
    params = await _plivo_params(request)
    call_uuid = params.get("CallUUID", "")
    what = (_transfer_targets.get(call_uuid) or {}).get("what", "human agent")
    status = params.get("DialStatus")
    print(f"[transfer] dial ended call={call_uuid} {what}: DialStatus={status} "
          f"DialHangupCause={params.get('DialHangupCause')} rang={params.get('DialRingStatus')}")
    response = plivoxml.ResponseElement()
    if status != "completed":
        response.add(plivoxml.SpeakElement(_UNAVAILABLE))
        response.add(plivoxml.HangupElement())
    return _xml(response)

@app.post("/plivo/hangup")
async def hangup(request: Request) -> Response:
    params = await _plivo_params(request)
    call_uuid = params.get("CallUUID", "")
    _pending_calls.pop(call_uuid, None)      # answered but the stream never started
    _transfer_targets.pop(call_uuid, None)
    print(f"[plivo] call {call_uuid} ended ({params.get('HangupCause')}, "
          f"{params.get('Duration')}s)")
    return Response(status_code=200)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Plivo REST: transfer / hang up a live call
# ---------------------------------------------------------------------------
def _rest_client() -> plivo.RestClient:
    return plivo.RestClient(require("PLIVO_AUTH_ID"), require("PLIVO_AUTH_TOKEN"))


async def _transfer_call(call_uuid: str, tag: str, session: Session) -> bool:
    """Points the live call's A-leg at /plivo/transfer (Plivo Transfer Call API).
    True if Plivo accepted it; the caller must then hang up the stream socket."""
    number = session.transfer_number or settings.HUMAN_AGENT_NUMBER
    what = session.handoff_summary or session.escalation_reason or "human agent"
    url = _public_url("/plivo/transfer") if settings.PUBLIC_BASE_URL else ""
    dialled = dial_number(number)
    line = f"{tag}[transfer] call={call_uuid} {what} number={_mask(dialled or number)} aleg_url={url or '-'}"
    problem = ("no valid Indian number to dial" if not dialled
               else "PLIVO_NUMBER (caller ID) is not a valid Indian number" if not dial_number(settings.PLIVO_NUMBER)
               else "PUBLIC_BASE_URL must be the public https URL" if not url.startswith("https://")
               or any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0")) else None)
    if problem:
        print(f"{line} -> NOT transferred: {problem}")
        return False
    _transfer_targets[call_uuid] = {"number": dialled, "what": what}
    try:
        result = await asyncio.to_thread(
            _rest_client().calls.transfer, call_uuid, legs="aleg",
            aleg_url=url, aleg_method="POST",
        )
    except Exception as exc:
        print(f"{line} -> Plivo transfer API error: {exc!r}")
        _transfer_targets.pop(call_uuid, None)
        return False
    print(f"{line} -> Plivo accepted: {getattr(result, 'message', result)!s} "
          f"(api_id={getattr(result, 'api_id', '-')})")
    return True


async def _hangup_call(call_uuid: str, tag: str) -> None:
    try:
        await asyncio.to_thread(_rest_client().calls.delete, call_uuid)
    except Exception as exc:          # the caller may already have hung up
        print(f"{tag}[plivo] hangup failed: {exc!r}")


# ---------------------------------------------------------------------------
# The audio stream: one WebSocket per call
# ---------------------------------------------------------------------------
class _PlivoCall:
    def __init__(self, ws: WebSocket, call_uuid: str) -> None:
        self.ws = ws
        self.call_uuid = call_uuid
        self.tag = f"[{call_uuid[:8]}] "
        self.audio = PhoneAudioStream()
        self._played_until = 0.0     # loop time when the queued audio finishes playing

    async def say(self, text: str, language: str = "en") -> None:
        """Streams the TTS audio to the caller and returns once they've heard it."""
        print(f"{self.tag}AI: {text}")
        loop = asyncio.get_running_loop()
        try:
            async for pcm in synthesize(text, language, _RATE):
                for i in range(0, len(pcm), _FRAME_BYTES):
                    piece = pcm[i:i + _FRAME_BYTES]
                    await self.ws.send_text(json.dumps({
                        "event": "playAudio",
                        "media": {
                            "contentType": "audio/x-l16",
                            "sampleRate": _RATE,
                            "payload": base64.b64encode(piece).decode(),
                        },
                    }))
                    self._played_until = (max(self._played_until, loop.time())
                                          + len(piece) / (_RATE * 2))
        except Exception as exc:
            print(f"{self.tag}(TTS failed: {exc})")
        remaining = self._played_until - loop.time()
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def run(self, session: Session) -> None:
        gate = TurnGate()
        self.audio.muted = True
        await self.say(opening_line(session), "en")
        self.audio.muted = False
        try:
            raw = stream_transcribe(config.get_sarvam_api_key(), settings.PHONE_STT_LANG,
                                    self.audio.generator())
            await converse(session, with_turn_gate(raw, gate), self.say, self.audio, gate, self.tag)
        except Exception as exc:          # STT/LLM outage: don't leave the caller in silence
            print(f"{self.tag}[plivo] call failed: {exc!r}")
            if self.audio.closed:         # caller already hung up
                return
            session.backend_failures += 1
            session.escalation_reason = "system_error"
            session.handoff_summary = build_handoff_summary(session, "system_error")
            session.set_stage(Stage.TRANSFERRED)
            await self.say(_TRANSFERRING["en"], "en")

        if session.stage == Stage.TRANSFERRED:
            print(f"{self.tag}[handoff] {session.handoff_summary}")
            if await _transfer_call(self.call_uuid, self.tag, session):
                # <Stream keepCallAlive="true"> blocks any new XML until the stream
                # disconnects, so the transfer only runs once we close our socket.
                # Closing it does NOT hang up: the call carries on to /plivo/transfer.
                try:
                    await self.ws.close()
                except Exception as exc:
                    print(f"{self.tag}[transfer] closing the stream socket failed: {exc!r}")
            else:                         # never leave the caller in dead air
                await self.say(_TRANSFER_FAILED, "en")
                await _hangup_call(self.call_uuid, self.tag)
        elif session.stage == Stage.COMPLETED:
            await _hangup_call(self.call_uuid, self.tag)


@app.websocket("/plivo/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    call = None
    session = None
    task = None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            event = msg.get("event")
            if event == "media" and call is not None:
                payload = (msg.get("media") or {}).get("payload")
                if payload:
                    call.audio.put(base64.b64decode(payload))
            elif event == "start" and call is None:
                start = msg.get("start") or {}
                call_uuid = start.get("callId") or msg.get("callId") or ""
                if call_uuid not in _pending_calls:
                    print(f"[plivo] refused stream for unknown call {call_uuid!r}")
                    await ws.close(code=1008)
                    return
                session = await store.create()
                session.caller_number = _pending_calls.pop(call_uuid) or None
                call = _PlivoCall(ws, call_uuid)
                print(f"{call.tag}[plivo] stream started, session {session.session_id}")
                task = asyncio.create_task(call.run(session))
            elif event == "stop":
                break
    except (WebSocketDisconnect, RuntimeError):   # RuntimeError: we closed it for a transfer
        pass
    finally:
        if call is not None:
            call.audio.close()            # ends STT, which ends the conversation loop
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=15)
            except Exception as exc:
                print(f"[plivo] call task ended with {exc!r}")
        if session is not None:
            if session.stage not in END_STAGES:
                session.set_stage(Stage.ABANDONED)
            try:
                await mongo.save_call_log(session)
            except Exception as exc:      # a logging failure must never break the server
                print(f"[call log] not saved: {exc!r}")
            await store.delete(session.session_id)