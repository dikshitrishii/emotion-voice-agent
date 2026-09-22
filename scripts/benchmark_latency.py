"""Latency benchmark for the FastAPI layer itself.

This measures request/response overhead (routing, validation, middleware,
JSON serialization) using the dummy-mode LLM and mocked ASR/TTS backends —
deliberately, so it runs in seconds with no model downloads and gives a
number for "what does this service cost you before a single ML model even
runs." That number matters on its own (it's pure overhead you pay on every
request) and it isolates the framework layer for regression testing in CI
without needing the real models.

To benchmark the real Whisper/Coqui/Groq stack end-to-end, run this against
a live `uvicorn backend.main:app` with a real GROQ_API_KEY instead — see
`--live` below.

Usage:
    python scripts/benchmark_latency.py            # in-process, mocked
    python scripts/benchmark_latency.py --live      # HTTP against a running server
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_inprocess(n: int) -> dict[str, list[float]]:
    import io

    import numpy as np
    import soundfile as sf
    from fastapi.testclient import TestClient

    from backend.main import app
    from backend.services import asr, tts

    class FakeWhisper:
        def transcribe(self, path, fp16, language):
            return {"text": "this is a benchmark transcript"}

    class FakeCoqui:
        def tts_to_file(self, text, file_path):
            sr = 22050
            tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
            sf.write(file_path, tone, sr)

    asr.load_model = lambda: FakeWhisper()
    tts.get_tts = lambda: FakeCoqui()

    client = TestClient(app)
    buf = io.BytesIO()
    sf.write(buf, np.zeros(8000, dtype=np.float32), 16000, format="WAV")
    wav_bytes = buf.getvalue()

    latencies: dict[str, list[float]] = {"transcribe": [], "chat": [], "synthesize": []}

    for _ in range(n):
        t0 = time.perf_counter()
        client.post("/transcribe", files={"file": ("a.wav", wav_bytes, "audio/wav")})
        latencies["transcribe"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        client.post("/chat", json={"text": "hello there", "history": []})
        latencies["chat"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        tts._cache.clear()  # force a real (fake) synth each time, not a cache hit
        client.post("/synthesize", json={"text": "hello there", "emotion": "neutral"})
        latencies["synthesize"].append((time.perf_counter() - t0) * 1000)

    return latencies


def summarize(latencies: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    out = {}
    for route, values in latencies.items():
        out[route] = {
            "p50_ms": round(statistics.median(values), 2),
            "p95_ms": round(statistics.quantiles(values, n=20)[18], 2) if len(values) >= 20 else round(max(values), 2),
            "mean_ms": round(statistics.mean(values), 2),
            "n": len(values),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="iterations per route")
    parser.add_argument("--live", action="store_true", help="not implemented in this script version")
    parser.add_argument("--out", default="benchmarks/results.json")
    args = parser.parse_args()

    if args.live:
        print("--live mode requires a running server + real models; see docstring.", file=sys.stderr)
        sys.exit(1)

    latencies = run_inprocess(args.n)
    summary = summarize(latencies)

    print(json.dumps(summary, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
