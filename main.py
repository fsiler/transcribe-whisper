#!/usr/bin/env python
import whisper
from sys import argv
import os
from datetime import timedelta

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{td.microseconds//1000:03d}"

print("loading model....", end="")
model = whisper.load_model("turbo")
print("done.")

for filename in argv[1:]:
    print(f"opening {filename}....")
    result = model.transcribe(filename, word_timestamps=True)

    # Create VTT filename
    vtt_filename = os.path.splitext(filename)[0] + ".vtt"

    with open(vtt_filename, "w", encoding="utf-8") as vtt_file:
        vtt_file.write("WEBVTT\n\n")

        for i, segment in enumerate(result["segments"]):
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            vtt_file.write(f"{i+1}\n")
            vtt_file.write(f"{start_time} --> {end_time}\n")
            vtt_file.write(f"{segment['text'].strip()}\n\n")

    print(f"VTT file created: {vtt_filename}")
