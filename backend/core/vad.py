"""Lightweight energy-based voice activity detection.

A full neural VAD (Silero, WebRTC) is the "correct" answer, but it's an
extra heavy dependency for a proxy problem: trimming the dead air a browser
mic capture always has at the start/end of a clip. A short-time energy
threshold does that in <5ms with numpy alone, cuts Whisper's input length
(and therefore latency) on typical clips, and gives us a clean seam to
swap in Silero later without touching any caller.
"""
from __future__ import annotations

import numpy as np


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 20,
    energy_threshold: float = 0.01,
    padding_ms: int = 100,
) -> np.ndarray:
    """Trim leading/trailing silence from a mono float32 waveform.

    Returns the original audio unchanged if the whole clip looks silent
    (avoids returning an empty array that would crash downstream ASR).
    """
    if audio.ndim != 1:
        audio = audio.mean(axis=-1)

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = max(1, len(audio) // frame_len)
    if n_frames == 0:
        return audio

    trimmed = audio[: n_frames * frame_len]
    frames = trimmed.reshape(n_frames, frame_len)
    energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

    voiced = np.where(energy > energy_threshold)[0]
    if voiced.size == 0:
        return audio  # looks entirely silent — let ASR decide, don't nuke input

    pad_frames = max(1, int(padding_ms / frame_ms))
    start = max(0, (voiced[0] - pad_frames) * frame_len)
    end = min(len(audio), (voiced[-1] + 1 + pad_frames) * frame_len)
    return audio[start:end]
