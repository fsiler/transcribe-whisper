#!/usr/bin/env python
import os
import time
import torch
import whisper
import subprocess
import json

from datetime import timedelta
from sys import argv

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{td.microseconds//1000:03d}"

def has_subtitle_stream(filename):
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', filename]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return any(stream['codec_type'] == 'subtitle' for stream in data['streams'])

def transcribe(filename):
    # Check if file has subtitle stream
    if has_subtitle_stream(filename):
        print(f"=== Skipping {filename} - Subtitle stream already exists.")
        return

    # Create output filename (MKA or MKV)
    output_filename = os.path.splitext(filename)[0] + ".mka"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16 = False if device=='cpu' else True

    print("loading model...", end="", flush=True)
    model = whisper.load_model("turbo").to(device)
    print("done.", flush=True)

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

    # Prepare SRT content
    srt_content = ""
    for i, segment in enumerate(result["segments"]):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        srt_content += f"{i+1}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n\n"

    # Write transcription to MKA/MKV file
    cmd = [
        'ffmpeg', '-i', filename,
        '-i', '-',
        '-map', '0', '-map', '1',
        '-c', 'copy',
        '-c:s', 'srt',
        output_filename
    ]

    subprocess.run(cmd, input=srt_content.encode('utf-8'), check=True)

    print(f"Transcription file created: {output_filename}")
    print(f"Audio length: {format_timestamp(audio_length)}")
    print(f"Transcription time: {format_timestamp(transcription_time)}")
    print(f"Transcribed at {ratio:.2f}x speed")

if __name__ == "__main__":
    for filename in argv[1:]:
        transcribe(filename)
