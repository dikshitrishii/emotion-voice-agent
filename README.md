#  Emotion-Aware Real-Time Voice AI Agent

A local-first, end-to-end voice AI agent that listens to your speech, understands context, detects the appropriate emotional tone, and responds with an emotion-aware synthesized voice.

> Started as a weekend ASR→LLM→TTS prototype; hardened into a service with tests, CI, structured logging, metrics, retries/fallbacks, rate limiting, and a VAD + conversation-memory layer — the parts that separate a demo from something you'd actually deploy.

---

## 📽️ Demo

```
You speak  →  Whisper transcribes  →  Groq LLM responds with emotion  →  Coqui TTS speaks back
```

**Example interaction:**

| You say | Agent responds | Emotion |
|---|---|---|
| "I just got the job offer!" | "That's absolutely wonderful news! You should be so proud of yourself." | 😊 happy |
| "I failed my exam today." | "I'm really sorry to hear that. It's okay to feel disappointed right now." | 🤗 empathetic |
| "What's the weather like?" | "I don't have live weather data, but I'd suggest checking your local forecast!" | 😐 neutral |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                       │
│   🎙 Mic Input  ─────────────────────  🔊 Audio Player      │
└────────────┬─────────────────────────────────┬─────────────┘
             │ POST /transcribe                │ audio bytes
             ▼                                  │
┌──────────────────────────────────────────────────────────┐
│                     FastAPI Backend                        │
│  request-id + timing middleware → rate limit → auth (opt.) │
│  /transcribe  →  /chat  →  /synthesize   /health  /metrics │
└────┬──────────────────┬───────────────────┬─────────────────┘
     ▼                  ▼                   ▼
  asr.py             llm.py              tts.py
  Whisper (local)    Groq LLaMA 3.3      Coqui TTS (local)
   + VAD trim         + retry/backoff     + concurrency-safe
   + size limits       + dummy fallback     tempfiles
                        + history summary   + LRU cache
     │                  │                   │
     ▼                  ▼                   ▼
 Transcript      {text, emotion,       WAV audio
                   used_fallback}
```

**What's new since the prototype:**
- **Resilience**: Groq calls retry with exponential backoff; if the API key is missing/invalid or the call still fails, the LLM service degrades to a local rule-based responder instead of a 500 (`used_fallback: true` in the response tells the caller which happened).
- **VAD**: energy-based silence trimming before Whisper — shrinks typical browser-mic clips and reduces both latency and hallucination-from-silence.
- **Bounded conversation memory**: history beyond `MAX_HISTORY_TURNS` is condensed into a rolling summary instead of growing the prompt unboundedly.
- **Observability**: JSON structured logs with a request ID threaded through every log line, Prometheus metrics at `/metrics`, a `/health` that reports per-component (ASR/LLM/TTS) status instead of a static `{"status": "ok"}`.
- **Traffic shaping**: optional bearer-token auth (`API_KEY`), per-route rate limiting (`slowapi`).
- **Concurrency-safety**: the original TTS service wrote every request to the same hardcoded `/tmp/tts_out.wav` — a real bug under concurrent load. Every request now gets its own tempfile; identical (text, emotion) pairs are served from an in-memory cache instead of re-synthesizing.
- **Non-blocking event loop**: Whisper/Coqui/Groq calls are blocking; they now run via `run_in_threadpool` so one slow request doesn't stall every other concurrent request.
- **Tests + CI**: 31 tests (unit + API integration), 87% coverage, ruff + mypy clean, GitHub Actions workflow.

---

## ⚙️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| ASR | [OpenAI Whisper](https://github.com/openai/whisper) (`base`) | Best-in-class free transcription, runs locally |
| VAD | Energy-based silence trim (numpy) | Trims dead air before ASR — no extra heavy dependency |
| LLM | [Groq + LLaMA 3.3 70B](https://console.groq.com), with local rule-based fallback | Free, extremely fast inference; degrades gracefully offline |
| TTS | [Coqui TTS](https://github.com/coqui-ai/TTS) | Local, free, no API key needed |
| Backend | [FastAPI](https://fastapi.tiangolo.com) + Pydantic v2 settings | Async, typed config, clean REST API |
| Resilience | [tenacity](https://github.com/jd/tenacity) (retry), [slowapi](https://github.com/laurentS/slowapi) (rate limit) | Standard, well-tested libraries over hand-rolled logic |
| Observability | [prometheus-client](https://github.com/prometheus/client_python), structured JSON logging | Queryable logs, scrapeable metrics |
| Frontend | [Streamlit](https://streamlit.io) | Rapid UI with built-in audio components |
| Testing | pytest, pytest-cov, ruff, mypy | Mocked at the model-loading boundary — full suite runs in ~1.5s, no GPU/downloads needed |

**100% free to run — no paid APIs required** (Groq's free tier is generous; the service also runs fully offline in dummy mode).

---

## 🗂️ Project Structure

```
emotion-voice-agent/
│
├── backend/
│   ├── main.py                 # FastAPI app, routes, middleware, exception handlers
│   ├── config.py                # Typed Settings (pydantic-settings) + dummy_mode logic
│   ├── core/
│   │   ├── exceptions.py        # Domain errors → HTTP status mapping
│   │   ├── logging.py           # JSON structured logging + request-id context
│   │   ├── security.py          # Optional API-key auth + rate limiter
│   │   └── vad.py               # Energy-based silence trimming
│   ├── services/
│   │   ├── asr.py               # Whisper transcription + VAD + validation
│   │   ├── llm.py               # Groq client + retry + dummy fallback
│   │   ├── tts.py                # Coqui synthesis + emotion prosody + LRU cache
│   │   └── memory.py            # Conversation history condensation/summarization
│   └── models/
│       └── schemas.py           # Pydantic request/response models with validation
│
├── frontend/
│   └── app.py                   # Streamlit UI (reads BACKEND_URL/API_KEY from env)
│
├── tests/                       # 31 tests, 87% coverage, all mocked — no model downloads
├── scripts/
│   └── benchmark_latency.py     # FastAPI-layer latency benchmark
├── benchmarks/
│   └── results.json / README.md
├── notebooks/
│   └── exploration.ipynb        # original ad-hoc ASR/LLM/TTS exploration notebook
│
├── .github/workflows/ci.yml     # lint + typecheck + test on every push
├── pyproject.toml               # ruff, mypy, pytest config
├── .env.example
├── requirements.txt             # runtime deps
├── requirements-dev.txt         # lightweight test deps (no torch/whisper/TTS)
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11
- macOS / Linux / WSL
- [Groq API key](https://console.groq.com) (free, no credit card) — **optional**, see dummy mode below
- `ffmpeg` installed (`brew install ffmpeg` on macOS)

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/emotion-voice-agent.git
cd emotion-voice-agent

python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
# macOS: install system dependency first
brew install ffmpeg espeak-ng

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Add your key to `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) — takes 2 minutes, no credit card. **Don't have one?** Leave `GROQ_API_KEY` unset (or `dummy-...`) — the service runs in **dummy mode**, using a local rule-based responder instead of hitting Groq. Everything else (ASR, TTS, the API itself) works identically; useful for offline dev, demos, and is exactly what the test suite runs against.

### 4. Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

First run downloads the Coqui TTS model (~150MB) and Whisper `base` model (~140MB); cached locally after that.

### 5. Run the frontend

```bash
streamlit run frontend/app.py
```

### 6. Open the app

`http://localhost:8501`.

---

## 🧪 Testing & CI

```bash
pip install -r requirements-dev.txt   # lightweight — no torch/whisper/TTS download
pytest                                 # 31 tests, ~1.5s, 87% coverage
ruff check backend tests               # lint
mypy backend                           # type check
```

Tests mock at the model-loading boundary (`asr.load_model`, `tts.get_tts`, `llm._call_groq`) rather than mocking HTTP calls one level down — this exercises real validation, real VAD math, real caching logic, and the real FastAPI request/response cycle, while never downloading a model or calling a paid API. `GROQ_API_KEY` isn't required in CI; tests run against dummy mode.

GitHub Actions (`.github/workflows/ci.yml`) runs lint + typecheck + tests on every push/PR.

---

## 📊 Benchmarks

`scripts/benchmark_latency.py` measures FastAPI-layer overhead (routing, validation, middleware, serialization) with models mocked — see [`benchmarks/README.md`](benchmarks/README.md) for the full explanation and latest numbers. It intentionally does *not* fabricate end-to-end latency for the real ML models; measure that yourself against a live server once real models are loaded (rough real-world numbers from the original prototype are kept below for reference).

Original prototype, tested on MacBook Air M1 (real models, real Groq call):

| Step | Latency |
|---|---|
| Whisper transcription (base) | ~1–2s |
| Groq LLM response | ~0.5–1s |
| Coqui TTS synthesis | ~3–5s |
| **Total round-trip** | **~5–8s** |

TTS remains the bottleneck. `tts_models/en/ljspeech/glow-tts` is ~2x faster at slight quality cost — change `COQUI_MODEL` in `.env`. Repeated (text, emotion) pairs now skip synthesis entirely via the in-memory cache.

---

## 🎛️ Configuration

All settings are typed and validated at startup (`backend/config.py`, `pydantic-settings`) — see `.env.example` for the full list. Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | unset | Unset or `dummy-*` → dummy mode (local fallback responder) |
| `MAX_AUDIO_MB` | 15 | Reject oversized `/transcribe` uploads with a clean 413 |
| `MAX_HISTORY_TURNS` | 8 | Turns kept verbatim before older history is summarized |
| `API_KEY` | unset | Unset = auth disabled (local dev). Set → all routes require `Authorization: Bearer <key>` |
| `RATE_LIMIT` | `30/minute` | Per-route request limit (slowapi) |
| `TTS_CACHE_SIZE` | 64 | Max (text, emotion) pairs cached in memory |

### Emotion → Voice Mapping

Coqui has no native emotion control, so emotion is approximated via speed/energy post-processing:

| Emotion | Speed | Energy | Feel |
|---|---|---|---|
| happy | 1.15x | 1.2x | Bright and upbeat |
| sad | 0.85x | 0.8x | Slow and soft |
| neutral | 1.0x | 1.0x | Natural |
| empathetic | 0.9x | 0.95x | Warm and measured |
| excited | 1.25x | 1.3x | Fast and energetic |
| calm | 0.9x | 0.9x | Gentle and steady |

---

## 🧩 API Reference

Base URL: `http://localhost:8000`. Every response carries `x-request-id` and `x-response-time-ms` headers; every error returns `{"error", "detail", "request_id"}`.

### `POST /transcribe`
`multipart/form-data`, field `file` (WAV/MP3/etc, ≤ `MAX_AUDIO_MB`).
```json
{ "text": "I am feeling very hot today.", "duration_s": 2.4 }
```

### `POST /chat`
```json
{ "text": "I am feeling very hot today.", "history": [{"role": "user", "content": "..."}] }
```
```json
{ "text": "I'm sorry to hear that. Try drinking some cool water.", "emotion": "empathetic", "used_fallback": false }
```
`used_fallback: true` means the Groq call failed/was unavailable and the local rule-based responder answered instead.

### `POST /synthesize`
```json
{ "text": "...", "emotion": "empathetic" }
```
```json
{ "audio_base64": "<base64 WAV>", "emotion": "empathetic", "cached": false }
```

### `GET /health`
```json
{
  "status": "ok",
  "version": "2.0.0",
  "dummy_mode": true,
  "components": {
    "asr": {"loaded": false, "detail": "base"},
    "tts": {"loaded": false, "detail": "tts_models/en/ljspeech/tacotron2-DDC"},
    "llm": {"loaded": true, "detail": "dummy_mode (rule-based fallback)"}
  }
}
```

### `GET /metrics`
Prometheus exposition format — `voice_agent_requests_total{route,status}`, `voice_agent_request_latency_seconds{route}`.

All endpoints accept an optional `Authorization: Bearer <API_KEY>` header (enforced only if `API_KEY` is set) and are individually rate-limited.

---

## 🐳 Docker

```bash
docker compose -f Docker-compose.yml up --build
```

Both images now run as non-root users and ship real `HEALTHCHECK`s (the frontend's was missing entirely before; the backend's existed in compose but silently failed — `curl` wasn't installed in the runtime image). Model weights persist across restarts via named volumes.

---

## 🧠 Design Decisions

**Why separate the three endpoints?**
Each service (ASR, LLM, TTS) can be swapped independently.

**Why a dummy-mode fallback instead of just erroring?**
A voice UI is unforgiving of dead air. If the Groq call fails — rate limited, network blip, revoked key — returning a 500 kills the conversation. Falling back to a rule-based reply (flagged via `used_fallback`) keeps the pipeline alive; the caller can decide how to surface that to the user.

**Why energy-based VAD instead of Silero/WebRTC VAD?**
Those are the "correct" answer for production-grade VAD, but they're an extra heavy dependency for a proxy problem: trimming the dead air a browser mic capture always has at the start/end. A numpy-only energy threshold solves that specific problem in <5ms and leaves a clean seam to swap in a neural VAD later without touching any caller.

**Why mock at the model-loading boundary in tests, not the HTTP layer?**
Mocking `asr.load_model()` / `tts.get_tts()` exercises the real FastAPI app, real Pydantic validation, real VAD math, and real caching logic — only the multi-GB model weights are swapped out. That's a meaningfully stronger test than mocking `requests.post`.

---

## 🚧 Roadmap

- [x] Conversation memory with summarization for long sessions
- [x] Docker compose setup (hardened: non-root, real healthchecks)
- [x] Voice activity detection (energy-based silence trimming)
- [ ] Streaming TTS (chunk audio as it generates, reduce perceived latency)
- [ ] Weather/time tool integration via function calling
- [ ] Whisper `large-v3` option for multilingual support
- [ ] WebSocket endpoint for lower-latency streaming interaction
- [ ] Neural VAD (Silero) swap-in behind the existing `core/vad.py` seam

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition
- [Groq](https://groq.com) — ultra-fast LLM inference
- [Coqui TTS](https://github.com/coqui-ai/TTS) — open-source text-to-speech
- [FastAPI](https://fastapi.tiangolo.com) — modern Python web framework
- [Streamlit](https://streamlit.io) — rapid ML app development
