import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WHISPER_MODEL = "base"
GROQ_MODEL= "llama-3.3-70b-versatile"

# Coqui TTS model — downloads once (~150MB), cached locally after
COQUI_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

# Emotion → prosody mapping for Coqui
# Coqui doesn't have native emotion control, so we use
# speech rate + energy as a proxy — passed via post-processing
EMOTION_STYLE_MAP = {
    "happy":      {"speed": 1.15, "energy": 1.2},
    "sad":        {"speed": 0.85, "energy": 0.8},
    "neutral":    {"speed": 1.0,  "energy": 1.0},
    "empathetic": {"speed": 0.9,  "energy": 0.95},
    "excited":    {"speed": 1.25, "energy": 1.3},
    "calm":       {"speed": 0.9,  "energy": 0.9},
}