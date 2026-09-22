import numpy as np
import pytest
import soundfile as sf

from backend.core.exceptions import AudioTooLargeError, InvalidAudioError
from backend.services import asr


def _make_wav_bytes(seconds: float = 1.0, sr: int = 16000) -> bytes:
    import io

    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(int(sr * seconds)) / sr).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV")
    return buf.getvalue()


def test_empty_audio_rejected():
    with pytest.raises(InvalidAudioError):
        asr.transcribe(b"")


def test_oversized_audio_rejected(monkeypatch):
    monkeypatch.setattr(asr, "get_settings", lambda: _FakeSettings(max_audio_mb=0.0001))
    with pytest.raises(AudioTooLargeError):
        asr.transcribe(_make_wav_bytes())


class _FakeSettings:
    def __init__(self, max_audio_mb):
        self.max_audio_mb = max_audio_mb


def test_transcribe_returns_stripped_text_and_duration(monkeypatch):
    class FakeModel:
        def transcribe(self, path, fp16, language):
            return {"text": "  hello world  "}

    monkeypatch.setattr(asr, "load_model", lambda: FakeModel())

    text, duration_s = asr.transcribe(_make_wav_bytes(seconds=1.0))

    assert text == "hello world"
    assert duration_s > 0
