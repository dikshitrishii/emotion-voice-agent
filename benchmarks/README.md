# Benchmarks

`results.json` is produced by `scripts/benchmark_latency.py` and measures
**FastAPI-layer overhead only** — routing, Pydantic validation, middleware,
JSON (de)serialization — with ASR/TTS mocked and the LLM in dummy mode.
It answers "what does this service cost before a single ML model runs,"
which is useful on its own (pure overhead paid on every request, real
regardless of which models you plug in) and runs in ~1 second in CI with
no model downloads.

Latest run (50 iterations/route, `RATE_LIMIT` raised to avoid throttling
the benchmark itself):

| Route | p50 | p95 | mean |
|---|---|---|---|
| `/transcribe` (mocked ASR) | 1.7ms | 2.4ms | 1.9ms |
| `/chat` (dummy-mode LLM) | 1.2ms | 1.6ms | 1.2ms |
| `/synthesize` (mocked TTS) | 2.1ms | 2.9ms | 2.2ms |

Regenerate with:

```bash
GROQ_API_KEY=dummy-test-key python scripts/benchmark_latency.py --n 50
```

**This is not end-to-end latency.** The real bottlenecks — Whisper
transcription, Coqui synthesis, the Groq round-trip — are the local ML
models and a network call, none of which this script runs (see the
"Heavy local models" note in the main README for why). To get real
end-to-end numbers, run the service normally with real models loaded and
time actual requests against `http://localhost:8000`.
