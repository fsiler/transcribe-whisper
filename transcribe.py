#!/usr/bin/env python
from sys import argv
import os
from datetime import timedelta
import time

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{td.microseconds//1000:03d}"

print("loading model....", end="")
print("done.")

for filename in argv[1:]:
    # Create VTT filename
    vtt_filename = os.path.splitext(filename)[0] + ".vtt"

    # Check if VTT file already exists
    if os.path.exists(vtt_filename):
        print(f"=== Skipping {filename} - VTT file already exists.")
        continue

    import torch
    import whisper
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16 = False if device=='cpu' else True
    model = whisper.load_model("turbo").to(device)

    print(f">>> opening {filename}....")

    # Get audio length
    audio = whisper.load_audio(filename)
    audio_length = len(audio) / whisper.audio.SAMPLE_RATE

    # Start timing the transcription
    start_time = time.time()

    result = model.transcribe(filename, word_timestamps=True, fp16=fp16)

    # End timing the transcription
    end_time = time.time()
    transcription_time = end_time - start_time

    # Calculate ratio
    ratio = audio_length / transcription_time

    with open(vtt_filename, "w", encoding="utf-8") as vtt_file:
        vtt_file.write("WEBVTT\n\n")

        for i, segment in enumerate(result["segments"]):
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            vtt_file.write(f"{i+1}\n")
            vtt_file.write(f"{start_time} --> {end_time}\n")
            vtt_file.write(f"{segment['text'].strip()}\n\n")

    print(f"VTT file created: {vtt_filename}")
    print(f"Audio length: {format_timestamp(audio_length)}")
    print(f"Transcription time: {format_timestamp(transcription_time)}")
    print(f"Transcribed at {ratio:.2f}x speed")

