#defining the schemas
from pydantic import BaseModel
from typing import Literal, Optional

EmotionType = Literal["happy", "sad", "neutral", "empathetic", "excited", "calm"]

class TranscribeResponse(BaseModel):
    text: str

class ChatRequest(BaseModel):
    text: str
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]

class ChatResponse(BaseModel):
    text: str
    emotion: EmotionType
    
class SynthesizeRequest(BaseModel):
    text: str
    emotion: EmotionType
    
class SynthesizeResponse(BaseModel):
    audio_base64: str          # base64-encoded WAV — easy to send over HTTP
    emotion: EmotionType