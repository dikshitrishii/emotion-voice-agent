import streamlit as st
import requests
import base64

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Emotion Voice Agent", page_icon="🎙️", layout="centered")
st.title("🎙️ Emotion-Aware Voice Agent")

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
            st.markdown(f"{EMOTION_EMOJI.get(emotion, '')} *{emotion}*")
        st.write(msg["content"])

# ── Play pending audio (rendered AFTER messages, BEFORE input) ──
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

        with st.spinner("Transcribing..."):
            resp = requests.post(
                f"{BACKEND_URL}/transcribe",
                files={"file": ("audio.wav", audio_bytes_raw, "audio/wav")},
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
            )
            if resp.status_code != 200:
                st.error(f"Chat failed: {resp.text}")
                st.stop()
            chat_data = resp.json()
            response_text = chat_data["text"]
            emotion = chat_data["emotion"]

        with st.spinner("Generating voice..."):
            resp = requests.post(
                f"{BACKEND_URL}/synthesize",
                json={"text": response_text, "emotion": emotion},
            )
            if resp.status_code != 200:
                st.error(f"TTS failed: {resp.text}")
                st.stop()
            audio_b64 = resp.json()["audio_base64"]

        # Save to history
        st.session_state.history.append({"role": "assistant", "content": response_text})
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "emotion": emotion
        })

        # Store audio in session state — renders on next rerun ABOVE the input widget
        st.session_state.pending_audio = audio_b64
        st.rerun()