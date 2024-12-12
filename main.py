#!/usr/bin/env python
import whisper
from sys import argv

print("loading model....", end="")
model = whisper.load_model("turbo")
print("done.")

for filename in argv[1:]:
  result = model.transcribe(filename, word_timestamps=True)

  for segment in result["segments"]:
      print(f"{filename}[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")

