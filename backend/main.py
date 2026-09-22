"""FastAPI application: /transcribe -> /chat -> /synthesize.

Hardening added over the original three-endpoint prototype:
  - request-ID + timing middleware, JSON structured logs
  - typed exception -> HTTP status mapping (no more bare 500s)
  - Prometheus metrics at /metrics
  - optional API-key auth + per-route rate limiting
  - blocking model calls (Whisper/Coqui/Groq) run in a threadpool so they
    don't stall the async event loop for other concurrent requests
  - a /health check that reports per-component readiness, not just "ok"
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool

from backend.config import get_settings
from backend.core.exceptions import VoiceAgentError
from backend.core.logging import configure_logging, get_logger, request_id_ctx
from backend.core.security import limiter, require_api_key
from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    ComponentStatus,
    HealthResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    TranscribeResponse,
)
from backend.services import asr, llm, tts

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

REQUEST_COUNT = Counter(
    "voice_agent_requests_total", "Total requests", ["route", "status"]
)
REQUEST_LATENCY = Histogram(
    "voice_agent_request_latency_seconds", "Request latency", ["route"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"starting up (dummy_mode={settings.dummy_mode}, environment={settings.environment})"
    )
    yield
    logger.info("shutting down")


app = FastAPI(title="Emotion Voice Agent", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    token = request_id_ctx.set(request_id)
    start = time.perf_counter()
    route = request.url.path
    try:
        response = await call_next(request)
    except Exception:
        REQUEST_COUNT.labels(route=route, status="500").inc()
        raise
    finally:
        request_id_ctx.reset(token)
    elapsed = time.perf_counter() - start
    REQUEST_LATENCY.labels(route=route).observe(elapsed)
    REQUEST_COUNT.labels(route=route, status=str(response.status_code)).inc()
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = f"{elapsed * 1000:.1f}"
    return response


@app.exception_handler(VoiceAgentError)
async def voice_agent_error_handler(request: Request, exc: VoiceAgentError):
    logger.warning(f"{type(exc).__name__}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "detail": exc.message,
            "request_id": request_id_ctx.get(),
        },
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "RateLimitExceeded", "detail": str(exc.detail), "request_id": request_id_ctx.get()},
    )


@app.get("/health", response_model=HealthResponse)
def health():
    components = {
        "asr": ComponentStatus(loaded=asr._model is not None, detail=settings.whisper_model),
        "tts": ComponentStatus(loaded=tts._tts is not None, detail=settings.coqui_model),
        "llm": ComponentStatus(
            loaded=True,
            detail="dummy_mode (rule-based fallback)" if settings.dummy_mode else settings.groq_model,
        ),
    }
    return HealthResponse(
        status="ok", version=app.version, dummy_mode=settings.dummy_mode, components=components
    )


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/transcribe", response_model=TranscribeResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit)
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    """Accepts an audio file, returns transcript text."""
    audio_bytes = await file.read()
    text, duration_s = await run_in_threadpool(asr.transcribe, audio_bytes)
    return TranscribeResponse(text=text, duration_s=duration_s)


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit)
async def chat_endpoint(request: Request, body: ChatRequest):
    """Accepts transcript + history, returns response text + emotion."""
    history = [m.model_dump() for m in body.history]
    result = await run_in_threadpool(llm.chat, body.text, history)
    return ChatResponse(
        text=result["text"], emotion=result["emotion"], used_fallback=result.get("used_fallback", False)
    )


@app.post("/synthesize", response_model=SynthesizeResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit)
async def synthesize_endpoint(request: Request, body: SynthesizeRequest):
    """Accepts text + emotion, returns base64-encoded audio."""
    audio_bytes, cached = await run_in_threadpool(tts.synthesize, body.text, body.emotion)
    audio_b64 = tts.audio_to_base64(audio_bytes)
    return SynthesizeResponse(audio_base64=audio_b64, emotion=body.emotion, cached=cached)
