import numpy as np

from backend.core.vad import trim_silence


def test_trims_leading_and_trailing_silence():
    sr = 16000
    silence = np.zeros(sr // 2, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    audio = np.concatenate([silence, tone, silence])

    trimmed = trim_silence(audio, sr)

    assert len(trimmed) < len(audio)
    assert len(trimmed) >= len(tone)  # padding keeps at least the voiced part


def test_all_silent_audio_returned_unchanged():
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    trimmed = trim_silence(audio, sr)
    assert len(trimmed) == len(audio)


def test_stereo_audio_is_downmixed():
    sr = 16000
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    stereo = np.stack([tone, tone], axis=-1)
    trimmed = trim_silence(stereo, sr)
    assert trimmed.ndim == 1
