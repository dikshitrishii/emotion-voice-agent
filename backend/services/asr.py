"""ASR (Automatic Speech Recognition) service — local Whisper transcription.

The model is loaded once at first use (lazy singleton), not per request:
cold load is ~5s, so loading it in the request path would put a 5s tax on
whichever unlucky user hits the service first *and* block every concurrent
request behind a global import lock. Loading lazily-but-once keeps startup
fast while still only paying the cost a single time.
"""
from __future__ import annotations

import io
import os
import tempfile
import time

import soundfile as sf

from backend.config import get_settings
from backend.core.exceptions import AudioTooLargeError, InvalidAudioError, TranscriptionError
from backend.core.logging import get_logger, log_extra
from backend.core.vad import trim_silence

logger = get_logger(__name__)

_model = None


def load_model():
    global _model
    if _model is None:
        import whisper  # imported lazily — heavy, and not needed for unit tests

        settings = get_settings()
        logger.info(f"loading whisper model: {settings.whisper_model}")
        _model = whisper.load_model(settings.whisper_model)
    return _model


def _validate_size(audio_bytes: bytes) -> None:
    settings = get_settings()
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > settings.max_audio_mb:
        raise AudioTooLargeError(
            f"Audio is {size_mb:.1f}MB, exceeds the {settings.max_audio_mb}MB limit"
        )
    if len(audio_bytes) == 0:
        raise InvalidAudioError("Empty audio payload")


def transcribe(audio_bytes: bytes) -> tuple[str, float]:
    """Transcribe raw audio bytes (WAV/MP3/etc). Returns (text, duration_s).

    Applies energy-based silence trimming before running Whisper, which
    shrinks typical browser-mic clips (long lead-in/trail-off silence) and
    reduces both latency and the chance Whisper hallucinates text from
    near-silent audio.
    """
    _validate_size(audio_bytes)
    model = load_model()

    # Decode straight from memory; only the (trimmed) clip Whisper reads
    # touches disk, instead of writing the upload and then a trimmed copy.
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean 422
        raise InvalidAudioError(f"Could not decode audio: {exc}") from exc

    trimmed = trim_silence(audio, sr)
    duration_s = round(len(trimmed) / sr, 3) if sr else 0.0

    fd, run_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(run_path, trimmed, sr)
        start = time.perf_counter()
        try:
            result = model.transcribe(run_path, fp16=False, language="en")
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        text = result["text"].strip()
        logger.info(
            "transcription complete",
            extra=log_extra(latency_ms=latency_ms, duration_s=duration_s, chars=len(text)),
        )
        return text, duration_s
    finally:
        os.unlink(run_path)
