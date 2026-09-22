import base64
import os

import requests
import streamlit as st

# Was hardcoded to localhost:8000 — silently ignored the BACKEND_URL env var
# docker-compose already sets for the containerized frontend. Now it's
# actually wired up, with a sensible local-dev default.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")  # optional — only needed if the backend enforces auth
REQUEST_TIMEOUT_S = 30

st.set_page_config(page_title="Emotion Voice Agent", page_icon="🎙️", layout="centered")
st.title("🎙️ Emotion-Aware Voice Agent")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


@st.cache_data(ttl=10)
def _backend_health() -> dict | None:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except requests.RequestException:
        return None


with st.sidebar:
    st.subheader("Backend status")
    health = _backend_health()
    if health is None:
        st.error("Backend unreachable")
    else:
        st.success(f"v{health['version']} · {health['status']}")
        if health.get("dummy_mode"):
            st.warning("LLM running in dummy mode (no Groq key) — replies are rule-based.")
        for name, comp in health.get("components", {}).items():
            st.caption(f"{name}: {'loaded' if comp['loaded'] else 'not yet loaded'} ({comp['detail']})")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
if "pending_audio" not in st.session_state:
    st.session_state.pending_audio = None  # stores base64 string to play

EMOTION_EMOJI = {
    "happy": "😊", "sad": "😢", "neutral": "😐",
    "empathetic": "🤗", "excited": "🤩", "calm": "😌"
}

# ── Conversation display ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            emotion = msg.get("emotion", "neutral")
            badge = f"{EMOTION_EMOJI.get(emotion, '')} *{emotion}*"
            if msg.get("used_fallback"):
                badge += "  ·  :grey[offline reply]"
            st.markdown(badge)
        st.write(msg["content"])

# ── Play pending audio ──
if st.session_state.pending_audio:
    audio_bytes = base64.b64decode(st.session_state.pending_audio)
    st.audio(audio_bytes, format="audio/wav")
    st.session_state.pending_audio = None

st.divider()

# ── Audio input ──
audio_input = st.audio_input("Speak your message")

if audio_input is not None:
    audio_bytes_raw = audio_input.read()
    audio_id = hash(audio_bytes_raw)

    if audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_id

        try:
            with st.spinner("Transcribing..."):
                resp = requests.post(
                    f"{BACKEND_URL}/transcribe",
                    files={"file": ("audio.wav", audio_bytes_raw, "audio/wav")},
                    headers=_headers(),
                    timeout=REQUEST_TIMEOUT_S,
                )
                if resp.status_code != 200:
                    st.error(f"Transcription failed: {resp.text}")
                    st.stop()
                transcript = resp.json()["text"]

            st.caption(f"📝 You said: *{transcript}*")

            st.session_state.history.append({"role": "user", "content": transcript})
            st.session_state.messages.append({"role": "user", "content": transcript})

            with st.spinner("Thinking..."):
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"text": transcript, "history": st.session_state.history[:-1]},
                    headers=_headers(),
                    timeout=REQUEST_TIMEOUT_S,
                )
                if resp.status_code != 200:
                    st.error(f"Chat failed: {resp.text}")
                    st.stop()
                chat_data = resp.json()
                response_text = chat_data["text"]
                emotion = chat_data["emotion"]
                used_fallback = chat_data.get("used_fallback", False)

            with st.spinner("Generating voice..."):
                resp = requests.post(
                    f"{BACKEND_URL}/synthesize",
                    json={"text": response_text, "emotion": emotion},
                    headers=_headers(),
                    timeout=REQUEST_TIMEOUT_S,
                )
                if resp.status_code != 200:
                    st.error(f"TTS failed: {resp.text}")
                    st.stop()
                audio_b64 = resp.json()["audio_base64"]
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")
            st.stop()

        # Save to history
        st.session_state.history.append({"role": "assistant", "content": response_text})
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "emotion": emotion,
            "used_fallback": used_fallback,
        })

        # Store audio in session state — renders on next rerun ABOVE the input widget
        st.session_state.pending_audio = audio_b64
        st.rerun()
