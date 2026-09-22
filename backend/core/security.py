"""API-key auth (optional) and rate limiting.

Auth is opt-in: if API_KEY is unset in the environment, the dependency is a
no-op so local dev/demo stays frictionless. Setting API_KEY switches every
protected route to requiring `Authorization: Bearer <key>`, which is what
you'd flip on before exposing this past localhost.
"""
from __future__ import annotations

from fastapi import Header, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.config import get_settings

limiter = Limiter(key_func=get_remote_address)


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.api_key:
        return  # auth disabled — open for local dev/demo

    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )
