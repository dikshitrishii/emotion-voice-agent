"""Pydantic request/response models.

Field constraints (max_length, min length via Field(...)) matter here
beyond documentation — they're the first line of defense against a
malformed or abusive request reaching a model call that costs real
latency/money, and they turn "the LLM choked on a 500KB prompt" into a
clean 422 at the edge instead.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EmotionType = Literal["happy", "sad", "neutral", "empathetic", "excited", "calm"]


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=4000)


class TranscribeResponse(BaseModel):
    text: str
    duration_s: float | None = Field(
        default=None, description="Audio duration after silence-trimming, in seconds"
    )


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=200)


class ChatResponse(BaseModel):
    text: str
    emotion: EmotionType
    used_fallback: bool = Field(
        default=False,
        description="True if the LLM call failed/was unavailable and a local rule-based reply was used instead",
    )


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    emotion: EmotionType


class SynthesizeResponse(BaseModel):
    audio_base64: str  # base64-encoded WAV — easy to send over HTTP
    emotion: EmotionType
    cached: bool = Field(default=False, description="True if served from the TTS cache")


class ComponentStatus(BaseModel):
    loaded: bool
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    dummy_mode: bool
    components: dict[str, ComponentStatus]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str
