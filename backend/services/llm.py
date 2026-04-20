from groq import Groq
import json
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a warm, emotionally intelligent voice assistant.

For EVERY response you must reply with valid JSON in this exact format:
{
  "text": "<your spoken response here>",
  "emotion": "<one of: happy, sad, neutral, empathetic, excited, calm>"
}

Rules:
- "text" should sound natural when spoken aloud. No markdown, no bullet points.
- "emotion" must reflect the tone of YOUR response (not the user's mood).
- Keep responses concise (2-4 sentences max) — this is a voice interface.
- Be warm, helpful, and human-sounding."""

def chat(user_text: str, history: list[dict]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=300,
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    if "text" not in parsed or "emotion" not in parsed:
        raise ValueError(f"LLM returned unexpected schema: {raw}")

    return parsed