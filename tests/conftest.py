"""Shared pytest fixtures.

Forces dummy_mode for the whole test session (no real Groq key needed) and
resets the cached Settings singleton so GROQ_API_KEY / API_KEY env vars set
by individual tests don't leak between them.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("GROQ_API_KEY", "dummy-test-key")
os.environ.setdefault("API_KEY", "")

from backend.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    return TestClient(app)
