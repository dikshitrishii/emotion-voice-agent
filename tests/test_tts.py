import numpy as np
import soundfile as sf

from backend.services import tts


class FakeTTS:
    def __init__(self):
        self.calls = 0

    def tts_to_file(self, text, file_path):
        self.calls += 1
        sr = 22050
        tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
        sf.write(file_path, tone, sr)


def test_synthesize_returns_wav_bytes_and_not_cached(monkeypatch):
    tts._cache.clear()
    fake = FakeTTS()
    monkeypatch.setattr(tts, "get_tts", lambda: fake)

    audio_bytes, cached = tts.synthesize("hello", "happy")

    assert isinstance(audio_bytes, bytes) and len(audio_bytes) > 0
    assert cached is False
    assert fake.calls == 1


def test_repeated_call_hits_cache(monkeypatch):
    tts._cache.clear()
    fake = FakeTTS()
    monkeypatch.setattr(tts, "get_tts", lambda: fake)

    tts.synthesize("same text", "calm")
    _, cached = tts.synthesize("same text", "calm")

    assert cached is True
    assert fake.calls == 1  # second call served from cache, model not invoked again


def test_different_emotion_is_a_cache_miss(monkeypatch):
    tts._cache.clear()
    fake = FakeTTS()
    monkeypatch.setattr(tts, "get_tts", lambda: fake)

    tts.synthesize("same text", "calm")
    _, cached = tts.synthesize("same text", "excited")

    assert cached is False
    assert fake.calls == 2
