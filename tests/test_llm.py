from backend.services import llm


def test_dummy_mode_returns_rule_based_reply_and_flags_it():
    result = llm.chat("I failed my exam today.", [])
    assert result["used_fallback"] is True
    assert result["emotion"] == "empathetic"
    assert isinstance(result["text"], str) and result["text"]


def test_dummy_mode_matches_happy_keyword():
    result = llm.chat("I just got the job offer!", [])
    assert result["emotion"] == "happy"


def test_dummy_mode_default_reply_for_unmatched_text():
    result = llm.chat("xyz completely unrelated gibberish", [])
    assert result["emotion"] == "neutral"


def test_groq_call_failure_falls_back_gracefully(monkeypatch):
    import os

    monkeypatch.setenv("GROQ_API_KEY", "gsk_real_looking_key_not_dummy")
    from backend.config import get_settings

    get_settings.cache_clear()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated Groq outage")

    monkeypatch.setattr(llm, "_call_groq", boom)

    result = llm.chat("hello there", [])
    assert result["used_fallback"] is True
    get_settings.cache_clear()
    os.environ["GROQ_API_KEY"] = "dummy-test-key"
