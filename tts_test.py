# test_tts.py — put this in your project root
from TTS.api import TTS
import soundfile as sf

print("Loading model...")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)
print("Model loaded. Synthesizing...")

tts.tts_to_file(text="Hello, this is a test.", file_path="test_output.wav")
print("Done. Check test_output.wav")