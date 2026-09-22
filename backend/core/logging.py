"""Structured (JSON) logging setup.

Plain-text logs are fine on a laptop; JSON logs are what actually gets
ingested by anything real (CloudWatch, Datadog, Loki). Emitting JSON from
day one means this service's logs are queryable/filterable without a
migration later, and every log line carries a request_id for tracing a
single request across the ASR -> LLM -> TTS pipeline.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # Quiet down noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "numba", "urllib3"):
        logging.getLogger(noisy).setLevel("WARNING")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_extra(**fields: object) -> dict[str, dict[str, object]]:
    """Usage: logger.info("msg", extra=log_extra(latency_ms=42))"""
    return {"extra_fields": fields}
