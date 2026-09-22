"""Typed application settings, loaded once from environment / .env.

Uses pydantic-settings so every config value is validated at startup
(wrong types or out-of-range values fail fast instead of surfacing as a
confusing error three requests later).
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── LLM (Groq) ──
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_timeout_s: float = Field(default=15.0, alias="GROQ_TIMEOUT_S")
    groq_max_retries: int = Field(default=3, alias="GROQ_MAX_RETRIES")

    # ── ASR (Whisper) ──
    whisper_model: str = Field(default="base", alias="WHISPER_MODEL")
    max_audio_mb: float = Field(default=15.0, alias="MAX_AUDIO_MB")

    # ── TTS (Coqui) ──
    coqui_model: str = Field(
        default="tts_models/en/ljspeech/tacotron2-DDC", alias="COQUI_MODEL"
    )
    tts_cache_size: int = Field(default=64, alias="TTS_CACHE_SIZE")

    # ── Conversation memory ──
    max_history_turns: int = Field(default=8, alias="MAX_HISTORY_TURNS")
    max_text_chars: int = Field(default=2000, alias="MAX_TEXT_CHARS")

    # ── API security / traffic shaping ──
    api_key: str | None = Field(default=None, alias="API_KEY")
    rate_limit: str = Field(default="30/minute", alias="RATE_LIMIT")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # ── Observability ──
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    @property
    def dummy_mode(self) -> bool:
        """True when no usable Groq key is configured.

        Rather than hard-failing every request, the LLM service falls back
        to a local rule-based responder. This keeps local dev, CI, and demos
        working without a real API key, and keeps the service degrading
        gracefully instead of returning 500s if the key is revoked/rate
        limited mid-conversation.
        """
        return not self.groq_api_key or self.groq_api_key.startswith("dummy")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Backwards-compatible module-level constants (existing code/tests import
# these names directly from backend.config).
settings = get_settings()
GROQ_API_KEY = settings.groq_api_key
GROQ_MODEL = settings.groq_model
WHISPER_MODEL = settings.whisper_model
COQUI_MODEL = settings.coqui_model

# Emotion -> prosody mapping for Coqui (no native emotion control, so we
# approximate with speech rate + energy as a post-processing proxy).
EMOTION_STYLE_MAP: dict[str, dict[str, float]] = {
    "happy": {"speed": 1.15, "energy": 1.2},
    "sad": {"speed": 0.85, "energy": 0.8},
    "neutral": {"speed": 1.0, "energy": 1.0},
    "empathetic": {"speed": 0.9, "energy": 0.95},
    "excited": {"speed": 1.25, "energy": 1.3},
    "calm": {"speed": 0.9, "energy": 0.9},
}
