#  Emotion-Aware Real-Time Voice AI Agent

A fully local, end-to-end voice AI agent that listens to your speech, understands context, detects the appropriate emotional tone, and responds with an emotion-aware synthesized voice — all using free, open-source tools.

> Built as a production-grade AI portfolio project demonstrating ASR, LLM integration, TTS, and real-time API design.

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
┌─────────────────────────────────────────────┐
│              Streamlit Frontend              │
│   🎙 Mic Input  ──────────  🔊 Audio Player  │
└────────────┬───────────────────────┬─────────┘
             │ POST /transcribe      │ audio bytes
             ▼                       │
┌─────────────────────────────────────────────┐
│              FastAPI Backend                 │
│  /transcribe  →  /chat  →  /synthesize       │
└────┬──────────────┬──────────────┬───────────┘
     ▼              ▼              ▼
  asr.py         llm.py         tts.py
  Whisper      Groq LLaMA     Coqui TTS
  (local)      (free API)      (local)
     │              │              │
     ▼              ▼              ▼
 Transcript    {text,emotion}   WAV audio
```

---

## ⚙️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| ASR | [OpenAI Whisper](https://github.com/openai/whisper) | Best-in-class free transcription, runs locally |
| LLM | [Groq + LLaMA 3.3 70B](https://console.groq.com) | Free API, extremely fast inference |
| TTS | [Coqui TTS](https://github.com/coqui-ai/TTS) | Local, free, no API key needed |
| Backend | [FastAPI](https://fastapi.tiangolo.com) | Async, fast, clean REST API |
| Frontend | [Streamlit](https://streamlit.io) | Rapid UI with built-in audio components |

**100% free to run — no paid APIs required.**

---

## 🗂️ Project Structure

```
emotion-voice-agent/
│
├── backend/
│   ├── main.py              # FastAPI app + all routes
│   ├── config.py            # Env vars, model settings, emotion map
│   ├── services/
│   │   ├── asr.py           # Whisper transcription service
│   │   ├── llm.py           # Groq LLM + emotion extraction
│   │   └── tts.py           # Coqui TTS synthesis
│   └── models/
│       └── schemas.py       # Pydantic request/response models
│
├── frontend/
│   └── app.py               # Streamlit UI
│
├── .env                     # API keys (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11
- macOS / Linux / WSL
- [Groq API key](https://console.groq.com) (free, no credit card)
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

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

Get your free Groq key at [console.groq.com](https://console.groq.com) — takes 2 minutes, no credit card.

### 4. Run the backend

```bash
# From project root
PYTHONPATH=backend uvicorn backend.main:app --reload --port 8000
```

First run will download the Coqui TTS model (~150MB) and Whisper base model (~140MB). This happens once and is cached locally.

### 5. Run the frontend

Open a new terminal tab:

```bash
source venv/bin/activate
streamlit run frontend/app.py
```

### 6. Open the app

Navigate to `http://localhost:8501` in your browser.

---

## 🎛️ Configuration

All settings live in `backend/config.py`:

```python
WHISPER_MODEL = "base"               # tiny / base / small / medium (larger = more accurate, slower)
GROQ_MODEL    = "llama-3.3-70b-versatile"
COQUI_MODEL   = "tts_models/en/ljspeech/tacotron2-DDC"
```

### Emotion → Voice Mapping

The agent maps LLM-detected emotions to TTS prosody settings:

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

Base URL: `http://localhost:8000`

### `POST /transcribe`

Transcribes audio to text using Whisper.

**Request:** `multipart/form-data`
```
file: <audio file> (WAV, MP3, etc.)
```

**Response:**
```json
{ "text": "I am feeling very hot today." }
```

---

### `POST /chat`

Sends transcript to LLM, returns response with emotion label.

**Request:**
```json
{
  "text": "I am feeling very hot today.",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "text": "I'm sorry to hear that. Try drinking some cool water.",
  "emotion": "empathetic"
}
```

---

### `POST /synthesize`

Converts text to emotion-aware speech audio.

**Request:**
```json
{
  "text": "I'm sorry to hear that. Try drinking some cool water.",
  "emotion": "empathetic"
}
```

**Response:**
```json
{
  "audio_base64": "<base64-encoded WAV>",
  "emotion": "empathetic"
}
```

---

### `GET /health`

```json
{ "status": "ok" }
```

---

## 🔄 How It Works

```
1. User records voice via browser mic (Streamlit audio_input)
   │
2. Audio bytes sent to POST /transcribe
   │  └─ Whisper model converts speech → text
   │
3. Transcript + conversation history sent to POST /chat
   │  └─ Groq LLaMA processes input
   │  └─ Returns JSON: { "text": "...", "emotion": "..." }
   │
4. Response text + emotion sent to POST /synthesize
   │  └─ Coqui TTS generates audio
   │  └─ Speed/energy adjusted based on emotion
   │  └─ Returns base64-encoded WAV
   │
5. Streamlit decodes audio and displays player
   └─ User clicks play and hears the emotion-aware response
```

---

## 🧠 Design Decisions

**Why separate the three endpoints?**
Each service (ASR, LLM, TTS) can be swapped independently. Want better TTS? Replace only `tts.py`. Want GPT-4? Replace only `llm.py`. No other code changes needed.

**Why JSON with `response_format: json_object`?**
Forces the LLM to always return valid, parseable JSON with `text` and `emotion` fields. Without this, LLMs occasionally wrap responses in markdown code fences which breaks `json.loads()`.

**Why Groq over local LLMs?**
Groq's free tier runs LLaMA 3.3 70B at ~500 tokens/second — faster than most local setups on consumer hardware. 14,400 requests/day free limit is more than enough for development and demos.

**Why Coqui over ElevenLabs?**
Completely free and local — no API key, no rate limits, no internet required after the first model download. Emotion is approximated via speed/energy parameters rather than neural style control.

---

## ⚡ Performance

Tested on MacBook Air M1:

| Step | Latency |
|---|---|
| Whisper transcription (base) | ~1–2s |
| Groq LLM response | ~0.5–1s |
| Coqui TTS synthesis | ~3–5s |
| **Total round-trip** | **~5–8s** |

The main bottleneck is Coqui TTS. To reduce it, switch to `tts_models/en/ljspeech/glow-tts` which is ~2x faster at slight quality cost — change `COQUI_MODEL` in `config.py`.

---

## 🔧 Troubleshooting

**`No module named 'backend'`**
```bash
# Always run from project root with PYTHONPATH set
PYTHONPATH=backend uvicorn backend.main:app --reload --port 8000
```

**Whisper install fails**
```bash
brew install ffmpeg
pip install openai-whisper
```

**Coqui TTS install fails on macOS**
```bash
brew install espeak-ng
pip install TTS
```

**Audio player appears but no sound**
Click the play button on the audio widget — browser autoplay policies require a manual click on macOS.

**Groq `RateLimitError`**
Free tier allows 14,400 requests/day and 30 requests/minute. Wait 60 seconds and retry.

---

## 🚧 Roadmap

- [ ] Conversation memory with summarization for long sessions
- [ ] Streaming TTS (chunk audio as it generates, reduce perceived latency)
- [ ] Weather/time tool integration via function calling
- [ ] Whisper `large-v3` option for multilingual support
- [ ] Docker compose setup for one-command deployment
- [ ] Voice activity detection (auto-detect when user stops speaking)

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