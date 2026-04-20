import base64
import io
import numpy as np
import soundfile as sf
from TTS.api import TTS
from config import COQUI_MODEL, EMOTION_STYLE_MAP

# Load once at import time — first run downloads the model (~150MB)
_tts = None

def get_tts():
    global _tts
    if _tts is None:
        print("[TTS] Loading Coqui model...")
        _tts = TTS(model_name=COQUI_MODEL, progress_bar=False)
    return _tts

def synthesize(text: str, emotion: str) -> bytes:
    """
    Synthesize speech and apply speed/energy adjustments per emotion.
    Returns WAV audio as bytes.
    """
    tts = get_tts()
    style = EMOTION_STYLE_MAP.get(emotion, EMOTION_STYLE_MAP["neutral"])

    # Coqui writes to a file or returns numpy — we use a BytesIO buffer
    buf = io.BytesIO()
    tts.tts_to_file(text=text, file_path="/tmp/tts_out.wav")

    # Load the generated audio
    audio, sample_rate = sf.read("/tmp/tts_out.wav")

    # Apply speed shift via resampling
    speed = style["speed"]
    if speed != 1.0:
        import librosa
        audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)

    # Apply energy (volume) scaling
    audio = audio * style["energy"]
    audio = np.clip(audio, -1.0, 1.0)  # prevent clipping distortion

    # Write final audio to buffer
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def audio_to_base64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")