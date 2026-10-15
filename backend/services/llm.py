"""LLM service — Groq LLaMA 3.3 70B with a local rule-based fallback.

Two production concerns this addresses that the original version didn't:

1. **Resilience**: a third-party API call is the single most likely thing
   to fail in this pipeline (rate limit, network blip, revoked key). It's
   wrapped in retry-with-backoff, and if it still fails (or no key is
   configured at all — `dummy_mode`), the service falls back to a small
   rule-based responder instead of a 500. Voice UIs are unforgiving of
   dead air; "a slightly generic reply" beats "the agent went silent".
2. **Bounded context**: conversation history is condensed via
   `services.memory` before every call, so a long-running session can't
   silently blow the model's context window.
"""
from __future__ import annotations

import json
import re

from tenacity import retry, retry_if_exception, wait_exponential

from backend.config import get_settings
from backend.core.exceptions import LLMError
from backend.core.logging import get_logger, log_extra
from backend.services.memory import condense_history, naive_summarize

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a warm, emotionally intelligent voice assistant.

For EVERY response you must reply with valid JSON in this exact format:
{
  "text": "<your spoken response here>",
  "emotion": "<one of: happy, sad, neutral, empathetic, excited, calm>"
}

Rules:
- "text" should sound natural when spoken aloud. No markdown, no bullet points.
- "emotion" must reflect the tone of YOUR response (not the user's mood).
- Keep responses concise (2-4 sentences max) — this is a voice interface.
- Be warm, helpful, and human-sounding."""

_client = None

# Very small keyword table for the offline/fallback responder. It's not
# trying to be a good chatbot — it's trying to keep the pipeline alive and
# demoable with zero external dependencies.
_FALLBACK_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\b(sad|failed|upset|depress|hurt|lost)\b", re.I),
     "empathetic", "I'm really sorry you're going through that. I'm here if you want to talk it through."),
    (re.compile(r"\b(happy|great|awesome|excited|got the job|won)\b", re.I),
     "happy", "That's wonderful to hear! You should be really proud of that."),
    (re.compile(r"\b(angry|furious|annoyed|frustrat)\b", re.I),
     "calm", "That sounds frustrating. Take a breath — want to talk through what happened?"),
    (re.compile(r"\b(weather|time|news)\b", re.I),
     "neutral", "I don't have live data for that right now, but I'm happy to help with anything else."),
]


def _get_client():
    global _client
    if _client is None:
        from groq import Groq  # imported lazily — not needed in dummy mode/tests

        settings = get_settings()
        # max_retries=0: the SDK retries twice by default, which stacked under
        # our own retry gave up to 9 attempts. Retries live only in _call_groq.
        _client = Groq(
            api_key=settings.groq_api_key, timeout=settings.groq_timeout_s, max_retries=0
        )
    return _client


def _fallback_reply(user_text: str) -> dict:
    for pattern, emotion, reply in _FALLBACK_RULES:
        if pattern.search(user_text):
            return {"text": reply, "emotion": emotion}
    return {
        "text": "Thanks for sharing that. Tell me a bit more and I'll do my best to help.",
        "emotion": "neutral",
    }


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient failures only.

    A 4xx from Groq (bad request, missing/retired model, invalid key) will
    fail identically on every attempt, so retrying just adds ~3s of backoff
    before the fallback. 408 (timeout) and 429 (rate limit) are the
    exceptions: those can succeed on a later attempt. Checked by duck-typing
    `status_code` so the groq SDK isn't needed at import time.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500:
        return status in (408, 429)
    return True


@retry(
    stop=lambda state: state.attempt_number >= get_settings().groq_max_retries,
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _call_groq(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool = True,
) -> str:
    client = _get_client()
    # Groq rejects json_object mode (400) unless the prompt mentions JSON, so
    # plain-text callers like summarize() must turn it off.
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra,
    )
    content = response.choices[0].message.content
    if not content:
        raise LLMError("Groq returned an empty response")
    return content


def summarize(messages: list[dict]) -> str:
    """LLM-backed summary of older turns; falls back to a heuristic summary."""
    settings = get_settings()
    if settings.dummy_mode or not messages:
        return naive_summarize(messages)
    try:
        prompt = [
            {"role": "system", "content": "Summarize this conversation snippet in one sentence."},
            {"role": "user", "content": json.dumps(messages)},
        ]
        raw = _call_groq(
            prompt, settings.groq_model, max_tokens=80, temperature=0.2, json_mode=False
        )
        return raw.strip()
    except Exception as exc:  # noqa: BLE001 - summarization is best-effort
        logger.warning(f"summary generation failed, using heuristic: {exc}")
        return naive_summarize(messages)


def chat(user_text: str, history: list[dict]) -> dict:
    settings = get_settings()
    bounded_history = condense_history(history, settings.max_history_turns, summarize)

    if settings.dummy_mode:
        logger.info("dummy_mode active — using rule-based responder", extra=log_extra())
        result = _fallback_reply(user_text)
        result["used_fallback"] = True
        return result

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *bounded_history,
                {"role": "user", "content": user_text}]

    try:
        raw = _call_groq(messages, settings.groq_model, max_tokens=300, temperature=0.7)
        parsed = json.loads(raw)
        if "text" not in parsed or "emotion" not in parsed:
            raise LLMError(f"LLM returned unexpected schema: {raw}")
        parsed["used_fallback"] = False
        return parsed
    except Exception as exc:  # noqa: BLE001 - any failure degrades to fallback, not a 500
        logger.warning(f"Groq call failed after retries, falling back: {exc}")
        result = _fallback_reply(user_text)
        result["used_fallback"] = True
        return result
