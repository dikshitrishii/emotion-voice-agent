from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import (
    TranscribeResponse, ChatRequest, ChatResponse,
    SynthesizeRequest, SynthesizeResponse
)
from services import asr, llm, tts

app = FastAPI(title="Emotion Voice Agent", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Accepts an audio file, returns transcript text."""
    try:
        audio_bytes = await file.read()
        text = asr.transcribe(audio_bytes)
        return TranscribeResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Accepts transcript + history, returns response text + emotion."""
    try:
        result = llm.chat(request.text, request.history)
        return ChatResponse(text=result["text"], emotion=result["emotion"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_endpoint(request: SynthesizeRequest):
    """Accepts text + emotion, returns base64-encoded audio."""
    try:
        audio_bytes = tts.synthesize(request.text, request.emotion)
        audio_b64 = tts.audio_to_base64(audio_bytes)
        return SynthesizeResponse(audio_base64=audio_b64, emotion=request.emotion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))