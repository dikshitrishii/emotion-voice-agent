# I have to generate the code for the ASR (Automatic Speech Recognition) service. This service will be responsible for converting audio input into text. I will use a popular Whisper Speech Recognition model to implement this functionality imported from config file.
import os, tempfile
# what is the purpose of tempfile module in python?The `tempfile` module in Python is used to create temporary files and directories. It provides a convenient way to generate temporary file names and manage temporary files without the need for manual cleanup. The module ensures that temporary files are created securely and are automatically deleted when they are no longer needed. This is particularly useful for tasks that require temporary storage of data, such as when processing audio files for ASR (Automatic Speech Recognition) services, where you might need to store audio data temporarily before processing it.

import whisper
# Model is loaded once at import time — not on every request
# This is critical for latency. Cold load takes ~5s; subsequent calls are fast.
_model = None
def load_model():
    global _model
    if _model is None:
        # choosing from config file
        from backend.config import WHISPER_MODEL
        print(f"[ASR] Loading Whisper model: {WHISPER_MODEL}")
        _model = whisper.load_model(WHISPER_MODEL)
    return _model

def transcribe(audio_bytes: bytes) -> str:
    """
    Accepts raw audio bytes (WAV/MP3/etc from the frontend),
    writes to a temp file, runs Whisper, returns transcript string.
    """
    model = load_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        temp.write(audio_bytes)
        tmp_path = temp.name

    try:
        result = model.transcribe(tmp_path, fp16=False, language="en")
        return result["text"].strip() # type: ignore
    finally:
        os.unlink(tmp_path)  # always clean up the temp file