"""Conversation memory management.

Sending the full transcript history on every turn is the naive approach —
it works for a 5-turn demo and then silently blows the LLM's context
window (and your token bill) on a real session. This keeps the last
`keep_recent` turns verbatim (they matter most for coherence) and folds
everything older into a single rolling summary line, so history length is
bounded regardless of how long the conversation runs.
"""
from __future__ import annotations

from backend.core.logging import get_logger

logger = get_logger(__name__)

Message = dict[str, str]


def condense_history(
    history: list[Message],
    keep_recent: int,
    summarize_fn,
) -> list[Message]:
    """Return a bounded history: [optional summary turn] + last N messages.

    `summarize_fn` is injected (rather than imported) so this stays unit
    testable without a live LLM call — tests pass a stub, production code
    passes the real Groq-backed summarizer.
    """
    if len(history) <= keep_recent:
        return history

    older = history[:-keep_recent]
    recent = history[-keep_recent:]

    summary_text = summarize_fn(older)
    logger.info(
        "condensed conversation history",
        extra={"extra_fields": {"dropped_turns": len(older), "kept_turns": len(recent)}},
    )
    summary_turn: Message = {
        "role": "system",
        "content": f"Earlier conversation summary: {summary_text}",
    }
    return [summary_turn, *recent]


def naive_summarize(messages: list[Message]) -> str:
    """Fallback summarizer used in dummy mode / when the LLM call fails.

    Not as good as an LLM-generated summary, but keeps the pipeline
    functional and deterministic (useful for tests) instead of hard-failing.
    """
    topics = [m["content"][:60] for m in messages if m.get("role") == "user"]
    if not topics:
        return "No prior topics."
    return "User previously discussed: " + "; ".join(topics)
