"""TTS service — local Coqui synthesis with emotion-driven prosody.

Two fixes over the original:

1. **Concurrency-safety**: the original wrote every request to a single
   hardcoded `/tmp/tts_out.wav` path. Two concurrent requests would clobber
   each other's audio file — a real bug under any real traffic, invisible
   in a single-user demo. Each call now gets its own tempfile.
2. **Caching**: identical (text, emotion) pairs — common in a demo/eval
   loop — are served from an in-memory LRU cache instead of re-running
   synthesis, which is the slowest stage in the pipeline (~3-5s).
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import time
from collections import OrderedDict

import numpy as np
import soundfile as sf

from backend.config import EMOTION_STYLE_MAP, get_settings
from backend.core.exceptions import SynthesisError
from backend.core.logging import get_logger, log_extra

logger = get_logger(__name__)

_tts = None
_cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()


def get_tts():
    global _tts
    if _tts is None:
        from TTS.api import TTS  # imported lazily — heavy, not needed for unit tests

        settings = get_settings()
        logger.info(f"loading Coqui TTS model: {settings.coqui_model}")
        _tts = TTS(model_name=settings.coqui_model, progress_bar=False)
    return _tts


def _cache_get(key: tuple[str, str]) -> bytes | None:
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_put(key: tuple[str, str], value: bytes) -> None:
    settings = get_settings()
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > settings.tts_cache_size:
        _cache.popitem(last=False)


def synthesize(text: str, emotion: str) -> tuple[bytes, bool]:
    """Synthesize speech with emotion-driven speed/energy. Returns (wav_bytes, cached)."""
    cache_key = (text, emotion)
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info("tts cache hit", extra=log_extra(emotion=emotion))
        return cached, True

    tts = get_tts()
    style = EMOTION_STYLE_MAP.get(emotion, EMOTION_STYLE_MAP["neutral"])

    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        start = time.perf_counter()
        try:
            tts.tts_to_file(text=text, file_path=out_path)
        except Exception as exc:  # noqa: BLE001
            raise SynthesisError(f"Coqui synthesis failed: {exc}") from exc

        audio, sample_rate = sf.read(out_path)

        speed = style["speed"]
        if speed != 1.0:
            import librosa

            audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)

        audio = audio * style["energy"]
        audio = np.clip(audio, -1.0, 1.0)  # prevent clipping distortion

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)
        result = buf.read()

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "synthesis complete",
            extra=log_extra(latency_ms=latency_ms, emotion=emotion, chars=len(text)),
        )

        _cache_put(cache_key, result)
        return result, False
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def audio_to_base64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")
