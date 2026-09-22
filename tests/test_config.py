import os

from backend.config import get_settings


def test_dummy_mode_true_when_key_missing(monkeypatch):
    # setenv to "" rather than delenv: on a machine with a real .env file,
    # delenv still leaves pydantic-settings free to fall back to the .env
    # file's value, which defeats the point of this test.
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    assert get_settings().dummy_mode is True
    os.environ["GROQ_API_KEY"] = "dummy-test-key"
    get_settings.cache_clear()


def test_dummy_mode_true_for_dummy_prefixed_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-anything")
    get_settings.cache_clear()
    assert get_settings().dummy_mode is True


def test_dummy_mode_false_for_real_looking_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real_key_123")
    get_settings.cache_clear()
    assert get_settings().dummy_mode is False


def test_cors_origin_list_parses_csv(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.com, https://b.com")
    get_settings.cache_clear()
    assert get_settings().cors_origin_list == ["https://a.com", "https://b.com"]
