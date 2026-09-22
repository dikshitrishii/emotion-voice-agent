import io

import numpy as np
import soundfile as sf


def _wav_bytes(seconds=0.5, sr=16000):
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(int(sr * seconds)) / sr).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV")
    return buf.getvalue()


def test_health_reports_dummy_mode(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["dummy_mode"] is True
    assert "asr" in body["components"] and "tts" in body["components"]


def test_metrics_endpoint_exposes_prometheus_text(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "voice_agent_requests_total" in resp.text


def test_chat_endpoint_dummy_mode(client):
    resp = client.post("/chat", json={"text": "I failed my exam today.", "history": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["emotion"] == "empathetic"
    assert body["used_fallback"] is True


def test_chat_rejects_empty_text(client):
    resp = client.post("/chat", json={"text": "", "history": []})
    assert resp.status_code == 422


def test_chat_rejects_oversized_text(client):
    resp = client.post("/chat", json={"text": "x" * 5000, "history": []})
    assert resp.status_code == 422


def test_transcribe_endpoint(client, monkeypatch):
    from backend.services import asr

    class FakeModel:
        def transcribe(self, path, fp16, language):
            return {"text": "hello from the test"}

    monkeypatch.setattr(asr, "load_model", lambda: FakeModel())

    resp = client.post(
        "/transcribe", files={"file": ("audio.wav", _wav_bytes(), "audio/wav")}
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "hello from the test"


def test_transcribe_rejects_empty_file(client):
    resp = client.post("/transcribe", files={"file": ("audio.wav", b"", "audio/wav")})
    assert resp.status_code == 422


def test_synthesize_endpoint(client, monkeypatch):
    from backend.services import tts

    class FakeTTS:
        def tts_to_file(self, text, file_path):
            sr = 22050
            tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
            sf.write(file_path, tone, sr)

    tts._cache.clear()
    monkeypatch.setattr(tts, "get_tts", lambda: FakeTTS())

    resp = client.post("/synthesize", json={"text": "hi there", "emotion": "neutral"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["emotion"] == "neutral"
    assert len(body["audio_base64"]) > 0


def test_response_has_request_id_and_timing_headers(client):
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
    assert "x-response-time-ms" in resp.headers


def test_api_key_enforced_when_configured(client, monkeypatch):
    import os

    from backend.config import get_settings

    monkeypatch.setenv("API_KEY", "secret123")
    get_settings.cache_clear()

    resp = client.post("/chat", json={"text": "hello", "history": []})
    assert resp.status_code == 401

    resp = client.post(
        "/chat",
        json={"text": "hello", "history": []},
        headers={"Authorization": "Bearer secret123"},
    )
    assert resp.status_code == 200

    os.environ["API_KEY"] = ""
    get_settings.cache_clear()
