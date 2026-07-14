import json
import queue

import ollama
import sounddevice as sd
from vosk import KaldiRecognizer, Model

DEVICE_INDEX = 6  # "default" (PipeWire) — set to the index printed by `sd.query_devices()`

model = Model("/opt/models/vosk-model-en-us-0.42-gigaspeech")
rec = KaldiRecognizer(model, 16000)
q = queue.Queue()


def callback(indata, frames, time, status):
    q.put(bytes(indata))


with sd.RawInputStream(device=DEVICE_INDEX, samplerate=16000, blocksize=8000,
                        dtype="int16", channels=1, callback=callback):
    print("Listening... (Ctrl+C to stop)")
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result())["text"]
            if text:
                print("You said:", text)
                response = ollama.chat(
                    model="gemma4:31b",
                    messages=[{"role": "user", "content": text}],
                )
                print("Ollama:", response["message"]["content"])
