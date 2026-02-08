import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import sounddevice as sd


def load_env(path: str = ".env") -> None:
	file_path = Path(__file__).resolve().parent / path
	if not file_path.exists():
		return
	for line in file_path.read_text().splitlines():
		line = line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		os.environ.setdefault(key.strip(), value.strip())


def synthesize_pcm(text: str) -> bytes:
	api_key = os.getenv("ELEVENLABS_API_KEY", "")
	voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
	model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
	if not api_key or not voice_id:
		raise RuntimeError("Missing ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID in .env")

	url = (
		f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
		"?output_format=pcm_16000"
	)
	payload = {
		"text": text,
		"model_id": model_id,
		"voice_settings": {
			"stability": 0.4,
			"similarity_boost": 0.75,
		},
	}
	data = json.dumps(payload).encode("utf-8")
	req = Request(
		url,
		data=data,
		method="POST",
		headers={
			"xi-api-key": api_key,
			"Content-Type": "application/json",
			"Accept": "audio/pcm",
		},
	)
	with urlopen(req, timeout=15) as resp:
		return resp.read()


def main() -> None:
	load_env()
	sentence = "Quick test: your form looks strong, keep breathing and stay tall."
	pcm = synthesize_pcm(sentence)
	audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
	sd.play(audio, samplerate=16000, blocking=True)
	time.sleep(0.1)


if __name__ == "__main__":
	main()
