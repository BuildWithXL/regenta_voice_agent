# Voice Agent — Hotel Receptionist AI

An AI voice agent pipeline for a hotel: it listens to a customer, understands
what they want, checks/updates a (currently fake) hotel backend, and replies
back in natural speech.

Two pipeline stages are built and working so far. The rest of the flow
(described in `Voice_Agent_Full_Flow_With_STT_TTS_Deepgram.md`) is still in
progress.

| Phase | What it does | Status |
|---|---|---|
| 1 — Speech → Text | Streams mic/phone audio to Deepgram Flux, returns interim + final transcripts | ✅ Built (`stt_client.py`, `audio_sources.py`) |
| 2 — Conversation memory | Stores conversation turns | ✅ Built (`session.py`) |
| 3 — AI understanding | Sends transcript to Groq LLM to interpret intent | ✅ Built (`ai_response.py`) |
| 4 — Tool picking | AI decides which backend action to call | ✅ Built (`ai_tools.py`) |
| 5 — Fake backend | Stand-in hotel booking system for testing | ✅ Built (stub logic inside `ai_tools.py`) |
| 6 — Reply generation | Turns backend results into natural replies | ✅ Built (part of `ai_response.py`) |
| 7 — Persistence | Confirmed bookings saved locally; MongoDB used **read-only** for lookups | ✅ Built (`local_booking_log.py`, `mongo.py`, `session.py`) |
| 8 — End-to-end test | Full pipeline test, start to finish | 🔶 In progress (`tests.py`) |
| 9 — Text → Speech | Deepgram Aura-2 turns replies into spoken audio | ✅ Built (`tts_client.py`) |
| — Phone integration (Plivo) | Real phone calls on +91 22 6423 2810 | ✅ Built (`plivo_server.py`) |

---

## Project layout

```
Voice_Agent/
├── main.py                 # CLI entry point — runs the live pipeline
├── config.py                # audio format, supported languages, env config
│
├── audio_sources.py         # microphone + WAV file streaming generators
├── stt_client.py            # Deepgram Flux streaming STT + TranscriptResult
│                            #   data class (that class used to be its own
│                            #   transcript_result.py — merged in, single user)
│
├── ai_response.py           # turns a transcript into an AI reply / action;
│                            #   calls Groq directly, runs tool calls, moves
│                            #   the session through its stages
├── ai_tools.py               # THE live tool layer: schema (TOOL_DEFINITIONS)
│                            #   + stub executors (EXECUTORS) for the 2 tools
│                            #   the AI can currently call — the "fake hotel
│                            #   backend" logic lives inline here, nowhere else
│
├── tts_client.py             # Deepgram Aura-2 text-to-speech
│
├── session.py                 # session data + storage — merged from what
│                            #   used to be two files, models.py (Session /
│                            #   Stage / Checklist dataclasses) and store.py
│                            #   (SessionStore, in-memory + asyncio.Lock)
├── mongo.py                   # MongoDB connection — READ-ONLY, no write/insert/
│                            #   update/delete calls exist in this file
├── local_booking_log.py      # where confirmed bookings actually get saved
│                            #   (append-only local JSONL, not Mongo)
│
├── tests.py                    # all tests — merged from what used to be
│                            #   test_ai_response.py + test_session.py
│
├── requirements.txt
└── Voice_Agent_Full_Flow_With_STT_TTS_Deepgram.md   # full pipeline design doc
```

> **MongoDB is strictly read-only.** `mongo.py` only defines `get_booking()`
> and `find_bookings()` — there is no insert/update/delete function anywhere
> in the file, so nothing in this app can write to whatever `MONGODB_URI` you
> provide. Confirmed bookings are persisted locally instead, in
> `bookings_log.jsonl` via `local_booking_log.py`. For real defense-in-depth,
> also create the MongoDB user you hand to this app with a **read-only**
> role on the Atlas/DB side — code-level restraint is enforced here, but a
> DB-side read-only role is the actual guarantee.

> Note: `venv`/site-packages folders are not part of the project — they're
> local virtual environments and should never be committed or zipped.

---

## 1. Deepgram setup (one-time)

1. Go to https://console.deepgram.com and create an account (no card
   required to start — new accounts get $200 in free credit).
2. Create an **API key** from the console. This single key covers both:
   - **STT** — Flux streaming transcription
   - **TTS** — Aura-2 text-to-speech
3. Create a `.env` file in the project root with:

   ```
   DEEPGRAM_API_KEY=your-deepgram-api-key
   DEEPGRAM_TTS_MODEL=aura-2-thalia-en   # or aura-2-luna-en, aura-2-stella-en
   GROQ_API_KEY=your-groq-api-key
   GROQ_MODEL=your-groq-model-name
   MONGODB_URI=your-mongodb-connection-string   # read-only use, see note above
   ```

---

## 2. Local setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then create `.env` with your real keys (see step 1).

`sounddevice` needs PortAudio installed at the OS level:

```bash
brew install portaudio           # macOS
sudo apt install libportaudio2   # Debian/Ubuntu
```

---

## 3. Run it

### Speech-to-text, from your microphone

```bash
python main.py --source mic --lang en
```

Speak, then pause — interim lines update in place, then a final line
prints with a confidence score.

Press **Ctrl+C** to stop.

### Speech-to-text, from a recorded file (no mic needed)

```bash
python main.py --source file --file path/to/customer_call.wav --lang en
```

Streamed in small chunks, paced in real time, so it behaves like a live
call. **Audio requirements:** 16-bit PCM WAV (`LINEAR16`), 16kHz, mono.
Convert first if needed:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 customer_call.wav
```

### Full pipeline test

```bash
python tests.py
```

Runs both test suites (AI response stage, then session storage) back to
back. Needs a real `GROQ_API_KEY`; the MongoDB-save check inside it is
expected to always print "SKIPPED" (see the note in `tests.py` — Mongo
is intentionally read-only, so there's no save function for it to call).

---

## 4. What you'll see on screen (STT)

```
Model: flux-general-en | Language: en
Listening... (Ctrl+C to stop)

[interim] hello i wanted to ask about my ord...
[interim] hello i wanted to ask about my order status
[FINAL ]  hello i wanted to ask about my order status   (confidence: 0.93)
  -> sent to next phase (NLU/intent): "hello i wanted to ask about my order status" [en, confidence=0.93]
```

- `[interim]` lines are "still typing" guesses — shown on screen only.
- `[FINAL]` lines are confirmed text — only these are passed on to the
  AI response stage (`ai_response.py`).

---

## 5. Language support

- **STT (Flux)**: English only right now, via `flux-general-en` in
  `stt_client.py`'s `FLUX_MODEL_MAP`. Add more languages by extending
  `FLUX_MODEL_MAP` and `config.SUPPORTED_LANGUAGES` — see
  [Deepgram Flux docs](https://developers.deepgram.com/docs/flux/quickstart).
- **TTS (Aura-2)**: English, Spanish, Dutch, French, German, Italian,
  Japanese. **No Hindi/Gujarati support** — if this project needs those,
  that's still an open decision, not solved by the current `tts_client.py`.

---

## 6. Known gotchas

- **TTS is REST-streaming, not WebSocket streaming.** Aura-2's WebSocket
  API isn't live yet, so `tts_client.py` streams the chunked REST HTTP
  response instead. Listen for gaps/stutters at chunk boundaries.
- **TTS sample rate assumption:** requests 24kHz linear16 audio. If
  pitch/speed sounds off, re-check Aura-2's current docs for the actual
  output rate.
- **Phone calls run through Plivo** (`plivo_server.py`); mic/file mode
  (`main.py`) is for local testing. See "Phone calls (Plivo)" below.
- **Fake backend only.** The stub logic inside `ai_tools.py` is a stand-in —
  no real hotel system is connected yet.

---

## 7. Definition of done — Phase 1 (STT) checklist

| Requirement | Where it's handled |
|---|---|
| Test without a real phone call (mic or file) | `audio_sources.py` (`MicrophoneStream`, `FileAudioStream`), via `--source` |
| Deepgram account + API key | Section 1 above, read via `DEEPGRAM_API_KEY` in `.env` |
| Streaming to Deepgram, results while still talking | `stt_client.py` uses `client.listen.v2.connect(...)` fed by a generator |
| Language selection: English | `--lang en`, via `config.SUPPORTED_LANGUAGES` and `stt_client.FLUX_MODEL_MAP` |
| Returns text, final/interim flag, confidence | `stt_client.py` yields `TranscriptResult(text, is_final, confidence, language_code)` |
| Only final text goes to next phase | `main.py` only forwards when `result.is_final` |
| Can tell final vs still-guessing | `is_final` flag, shown as `[FINAL]` vs `[interim]` |

## Definition of done — Phase 9 (TTS) checklist

Confirm by ear, using `tts_client.py` directly (e.g. from a Python shell)
or by listening to replies generated through the live pipeline:
- clear, natural-sounding English speech
- correct pronunciation of numbers/prices/dates
- audio starts playing quickly, not after a long wait
- warm tone, not robotic

---

## Roadmap

1. Finish end-to-end testing (Phase 8) — `tests.py`
2. Connect a real hotel backend in place of the stub logic in `ai_tools.py`
3. Real phone calls: done with Plivo (`plivo_server.py`)
4. Decide on Hindi/Gujarati support strategy for TTS
5. Investigate Aura-2 WebSocket streaming once available, to replace the
   current REST-streaming workaround in `tts_client.py`
# Hotel customer-support voice agent

Phone agent for existing bookings: it verifies the caller (booking ID + email),
answers hotel questions, cancels bookings, opens support tickets and hands over to
a human when needed. English and Hindi. Speech: Sarvam (STT + TTS). LLM: Groq.
Data: MongoDB.

## Setup
    pip install -r requirements.txt
    cp .env.example .env        # fill in GROQ_API_KEY, SARVAM_API_KEY, MONGODB_URI
    # fill in hotel_info.json (empty answers = "unknown", the agent never guesses)

## Run
    python main.py --source mic --lang en
    python main.py --source file --file sample.wav --lang hi

## Tests (offline, no API keys needed)
    python test_support.py

## Files
| File | Role |
|---|---|
| main.py | Call loop: audio in, STT, reply, TTS, handoff, call log |
| ai_response.py | LLM loop, tool guard rails, escalation, handoff summary |
| support_tools.py | Tickets, hotel FAQ (`getHotelInfo`), `transferToHuman` |
| booking_tools.py | `getBookingDetails` (verification), `cancelBooking` |
| escalation.py | Rules for when to hand over (EN / HI / romanized Hindi) |
| guardrails.py | Prompt-injection, tool-data sanitizing and grounded-reply checks |
| session.py, models.py | Call state and ticket model |
| mongo.py | Bookings, tickets (`supportTickets`), call logs (`supportCalls`) |
| config.py, .env.example | Settings (`STT_MIN_CONFIDENCE`, `MAX_FAILED_VERIFICATIONS`, ...) |
| audio_sources.py, sarvam_*_client.py, language_utils.py | Audio, speech and text helpers |

## Safety rules (enforced in code)
- Booking lookups only use an ID and email the caller actually said; after
  `MAX_FAILED_VERIFICATIONS` wrong attempts a human takes over.
- Cancelling needs a verified booking, an explicit "yes", and a stay that has not started.
- Ticket tools need a verified booking; only one ticket per call, only after a "yes".
- Payment disputes, legal threats, fraud and requests for a person transfer immediately.

## Guardrails (guardrails.py, wired in ai_response.py)
- **Prompt injection (input):** caller speech is normalised (zero-width and full-width
  tricks removed, capped at `MAX_INPUT_CHARS`) and checked in English, Hindi and
  romanized Hindi. A flagged turn never reaches the model, is stored as a placeholder,
  gets a fixed reply, and `MAX_INJECTION_ATTEMPTS` of them hand the call to a human.
- **Indirect injection (data):** everything read from MongoDB or `hotel_info.json` is
  sanitized before the model sees it, and fetched data is wrapped in `<fetched_data>`
  in the prompt and labelled as data, not instructions.
- **Knowledge-based grounding (output):** each reply is checked against this call's
  tool results and what the caller said. Booking/ticket IDs, emails, phone numbers,
  amounts and clock times must be grounded; hotel times and policies (breakfast, pets,
  parking, free cancellation) need a matching `getHotelInfo` answer; refund promises
  are blocked; internal tool names and prompt text can't be spoken. A blocked reply is
  regenerated once, then replaced with a safe line that offers a human.
- **Limits:** `MAX_TOOL_CALLS_PER_CALL`; ticket text is cleaned before it is stored.
- Write `hotel_info.json` values the way they should be spoken ("2 PM", not "14:00"),
  because times are matched literally.
- Blocks and removals are counted per call (`guardrail_blocks`, `injection_attempts`
  in `supportCalls`): review them to tune the patterns.

## Still to do
- Real call transfer: `_finish_call` in main.py has a TODO for your telephony provider.
- Fill `hotel_info.json` (per-hotel entries go under `hotels`).
- The `supportCalls` collection stores transcripts (names, emails): restrict access to it.

---

## Phone calls (Plivo)

Number: **+91 22 6423 2810**. `plivo_server.py` answers it: Plivo streams the
call audio (PCM16, 16 kHz, both ways) over a WebSocket, and each caller turn
goes through the same loop as mic mode (`conversation.py`).

| Endpoint | Used by |
|---|---|
| `POST /plivo/answer` | Plivo Answer URL: returns `<Stream>` XML |
| `WS /plivo/stream` | the call audio |
| `POST /plivo/hangup` | Plivo Hangup URL |
| `POST /plivo/transfer` | `<Dial>` to `HUMAN_AGENT_NUMBER` when the agent hands over |
| `GET /health` | uptime checks |

**Settings** (in `.env`): `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`,
`PLIVO_NUMBER`, `PUBLIC_BASE_URL`, `HUMAN_AGENT_NUMBER`, `PHONE_STT_LANG`.

**Local test**

```bash
pip install -r requirements.txt
uvicorn plivo_server:app --port 8000
ngrok http 8000                    # set PUBLIC_BASE_URL to the https URL it prints, restart uvicorn
```

Plivo console -> Voice -> Applications -> Add New Application:
Answer URL `<PUBLIC_BASE_URL>/plivo/answer` (POST), Hangup URL
`<PUBLIC_BASE_URL>/plivo/hangup` (POST). Then Phone Numbers -> +91 22 6423 2810
-> Application: that app. Call the number.

For the production server, see "Deploying on AWS" below.

---

## Deploying on AWS

The agent runs on the existing Hotellifyr AWS server, next to the other
Hotellifyr apps:

| | |
|---|---|
| Server | `3.23.121.245` (Ubuntu, user `ubuntu`, login with the `.pem` key) |
| Code folder | `/var/www/New_Products/Hotellifyr/hotellifyr_voice_agent` |
| App | `uvicorn plivo_server:app` on `127.0.0.1:9110`, kept running by **systemd** |
| Web server | **Apache** (already serving the other sites) forwards to the app |
| Public URL | `https://voice.hotellifyr.com` (Let's Encrypt certificate, via snap certbot) |
| Phone number | Plivo **+91 22 6423 2810** |
| Keys | all in `.env` in the code folder, including the MongoDB URL |

```
Caller -> Plivo (+91 22 6423 2810) -> https://voice.hotellifyr.com (Apache, port 443)
          -> 127.0.0.1:9110 (uvicorn plivo_server:app, systemd service "voice-agent")
          -> Sarvam (speech), Groq (AI), MongoDB (data)
```

Plivo only connects to `https://` and `wss://` addresses on port 443, so
Plivo never talks to port 9110 directly; Apache does.

### One-time setup

#### Step 1: AWS and DNS (already done)

- DNS: an **A record** `voice.hotellifyr.com` -> `3.23.121.245`. Note the
  double "l" in hotellifyr. Check it with `dig +short voice.hotellifyr.com`.
- Security group: ports **80** and **443** open to everyone (Apache), port
  **22** open to your office IP (SSH). Port 9110 does **not** need to be
  public, since Apache reaches it inside the server.

#### Step 2: Log in

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@3.23.121.245
```

#### Step 3: Get the code and install Python packages

The server needs Python 3.10 or newer (`python3 --version`).

```bash
cd /var/www/New_Products/Hotellifyr
git clone https://github.com/hotellifyr-global/hotellifyr_voice_agent.git
cd hotellifyr_voice_agent

sudo apt install -y python3-venv          # only if "python3 -m venv" fails
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

If git asks for a password, use a GitHub personal access token.

#### Step 4: Create `.env`

```bash
cp .env.example .env
nano .env
chmod 600 .env
```

Fill in every empty value (no quotes, no spaces around `=`):

| Key | Value |
|---|---|
| `GROQ_API_KEY` | from console.groq.com |
| `SARVAM_API_KEY` | from dashboard.sarvam.ai |
| `MONGODB_URI` | MongoDB runs on this same server, so use `127.0.0.1`: `mongodb://USER:PASSWORD@127.0.0.1:27017/hotellifyr` |
| `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN` | Plivo console home page |
| `PUBLIC_BASE_URL` | `https://voice.hotellifyr.com` (exactly this: https, no trailing slash) |
| `HUMAN_AGENT_NUMBER` | phone that receives transfers, with country code, e.g. `+919800000000` |

`.env` is in `.gitignore`. Never commit it, and never put real keys in
`.env.example`, which is committed.

#### Step 5: Run the app forever (systemd)

```bash
sudo nano /etc/systemd/system/voice-agent.service
```

```ini
[Unit]
Description=Voice agent (Plivo phone server)
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/var/www/New_Products/Hotellifyr/hotellifyr_voice_agent
ExecStart=/var/www/New_Products/Hotellifyr/hotellifyr_voice_agent/venv/bin/uvicorn plivo_server:app --host 127.0.0.1 --port 9110 --proxy-headers
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now voice-agent
sleep 8
curl http://127.0.0.1:9110/health          # {"status":"ok"}
```

`Restart=always` restarts it within 3 seconds after a crash, and `enable`
starts it on every boot. It keeps running after you log out. Don't start it
by hand with `uvicorn`, `nohup` or `screen` (and don't also add it to PM2),
or two copies will fight over port 9110.

#### Step 6: Apache site for voice.hotellifyr.com

Turn on the proxy modules (safe for the other sites):

```bash
sudo a2enmod proxy proxy_http proxy_wstunnel headers rewrite ssl
```

```bash
sudo nano /etc/apache2/sites-available/voice-agent.conf
```

```apache
<VirtualHost *:80>
    ServerName voice.hotellifyr.com

    ProxyPreserveHost On
    ProxyTimeout 3600

    # Call audio (WebSocket) - must stay ABOVE the "/" lines
    ProxyPass        /plivo/stream ws://127.0.0.1:9110/plivo/stream
    ProxyPassReverse /plivo/stream ws://127.0.0.1:9110/plivo/stream

    # Plivo webhooks and /health
    ProxyPass        / http://127.0.0.1:9110/
    ProxyPassReverse / http://127.0.0.1:9110/

    ErrorLog ${APACHE_LOG_DIR}/voice-agent-error.log
    CustomLog ${APACHE_LOG_DIR}/voice-agent-access.log combined
</VirtualHost>
```

```bash
sudo a2ensite voice-agent
sudo apache2ctl configtest                 # must say "Syntax OK" before reloading
sudo systemctl reload apache2
curl http://voice.hotellifyr.com/health    # {"status":"ok"}
```

Do not install Nginx on this server: Apache already uses ports 80 and 443.

#### Step 7: HTTPS certificate (snap certbot)

Use the **snap** certbot. The Ubuntu (apt) certbot is broken on this server
(see Troubleshooting).

```bash
sudo apt remove -y certbot python3-certbot-apache   # "remove" keeps /etc/letsencrypt
sudo snap install core
sudo snap install --classic certbot
sudo ln -sf /snap/bin/certbot /usr/bin/certbot

sudo certbot --apache -d voice.hotellifyr.com
# choose: redirect HTTP to HTTPS

curl https://voice.hotellifyr.com/health   # {"status":"ok"}
sudo certbot renew --dry-run               # auto-renewal check
```

Certbot creates `/etc/apache2/sites-available/voice-agent-le-ssl.conf` (the
same proxy lines on port 443) and renews the certificate automatically. It
also renews the other sites' Let's Encrypt certificates in `/etc/letsencrypt`.

#### Step 8: Connect the Plivo number

1. Plivo console -> **Voice -> Applications -> Add New Application**:
   - Name: `voice-agent`
   - Answer URL: `https://voice.hotellifyr.com/plivo/answer`, method **POST**
   - Hangup URL: `https://voice.hotellifyr.com/plivo/hangup`, method **POST**
2. **Phone Numbers -> Your Numbers -> +91 22 6423 2810** -> Application Type
   **XML Application** -> select `voice-agent` -> Update.

#### Step 9: Test a real call

```bash
sudo journalctl -u voice-agent -f
```

Call +91 22 6423 2810. You should see `[plivo] incoming call`, then
`stream started`, then each `You:` / `AI:` turn. Check:

- [ ] a new booking, start to finish (it appears in MongoDB `bookings`)
- [ ] looking up an existing booking and cancelling it
- [ ] a call in Hindi
- [ ] asking for a human: the call is transferred to `HUMAN_AGENT_NUMBER`
- [ ] hanging up mid-call: the call log is still saved

### Deploying a new version

```bash
ssh -i your-key.pem ubuntu@3.23.121.245
cd /var/www/New_Products/Hotellifyr/hotellifyr_voice_agent
git pull
venv/bin/pip install -r requirements.txt
sudo systemctl restart voice-agent
curl http://127.0.0.1:9110/health
```

Calls in progress are cut when the app restarts, so deploy when it's quiet.
After changing only `.env`: `sudo systemctl restart voice-agent`.

The hotel knowledge base (`hotel_kb/`) is read from the code folder. After
regenerating it with `kb_build.py`, commit it and deploy as above.

### Useful commands

| What | Command |
|---|---|
| Live app logs | `sudo journalctl -u voice-agent -f` |
| Last 100 log lines | `sudo journalctl -u voice-agent -n 100 --no-pager` |
| Restart / stop / start | `sudo systemctl restart voice-agent` (or `stop` / `start`) |
| Status | `sudo systemctl status voice-agent --no-pager` |
| Apache errors for this site | `sudo tail -50 /var/log/apache2/voice-agent-error.log` |
| Is port 9110 listening? | `sudo ss -ltnp \| grep 9110` |
| Certificate status | `sudo certbot certificates` |

### Troubleshooting

| Symptom | Fix |
|---|---|
| `curl 127.0.0.1:9110` -> `Connection refused`, status shows `status=203/EXEC` | The service file points to the wrong folder. `WorkingDirectory` and `ExecStart` must use `/var/www/New_Products/Hotellifyr/hotellifyr_voice_agent`, and `venv/bin/uvicorn` must exist there (Step 3). Then `daemon-reload` and `restart`. |
| Service keeps restarting (`activating (auto-restart)`) | `sudo journalctl -u voice-agent -n 50 --no-pager` shows the Python error, usually a missing value in `.env` or a missing package (`venv/bin/pip install -r requirements.txt`). |
| `certbot` crashes with `AttributeError: module 'lib' has no attribute 'GEN_EMAIL'` | The apt certbot clashes with a pip-installed `cryptography` on this server. Use the snap certbot (Step 7). Don't uninstall `cryptography`; other apps may need it. |
| `/etc/nginx/sites-available: No such file or directory` | This server uses Apache, not Nginx. Use Step 6. |
| `503 Service Unavailable` from Apache | The app isn't running: `sudo systemctl status voice-agent`. |
| Call says "application error" or drops at once | Check the logs. Are the Plivo Answer/Hangup URLs exactly as in Step 8, with POST? |
| `403 invalid Plivo signature` in the logs | `PUBLIC_BASE_URL` must be exactly `https://voice.hotellifyr.com`, and `PLIVO_AUTH_TOKEN` must be the current token. |
| Call connects but there's silence | Look for `Sarvam` or `Groq` errors in the logs. Check that the `/plivo/stream` lines are in **both** `voice-agent.conf` and `voice-agent-le-ssl.conf`, and that `proxy_wstunnel` is enabled. |
| `ServerSelectionTimeoutError` (MongoDB) | Check `MONGODB_URI` in `.env`; with MongoDB on this server, the host is `127.0.0.1`. |
| Transfer doesn't ring anyone | `HUMAN_AGENT_NUMBER` must be set, with country code. |

### Good to know

- **Run one copy only.** Call sessions are kept in memory, so a second copy
  (or a second server) would lose track of calls.
- **Rotate leaked keys.** If a key ever lands in git, regenerate it at its
  provider, update `.env` on the server and restart the service.
- **Latency.** The server is in an AWS US region; each reply reaches Indian
  callers a little later than it would from Mumbai (ap-south-1).
