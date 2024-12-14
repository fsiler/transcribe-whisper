#!/usr/bin/env python
import os
import time
import torch
import whisper
import subprocess
import json
import shutil

from datetime import timedelta
from sys import argv

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{td.microseconds//1000:03d}"

def get_file_streams(filename):
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', filename]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return data['streams']

def has_subtitle_stream(streams):
    return any(stream['codec_type'] == 'subtitle' for stream in streams)

def has_only_audio_and_subtitles(streams):
    return all(stream['codec_type'] in ['audio', 'subtitle'] for stream in streams)

def set_aside_original_file(filename):
    temp_filename = filename + '.temp'
    shutil.move(filename, temp_filename)
    return temp_filename

def transcribe(filename):
    print(f"=== examining {filename}: ", end="", flush=True)
    streams = get_file_streams(filename)

    if has_subtitle_stream(streams):
        print(f"file has subtitle stream, skipping.")
        return

    original_filename = filename

    # If the source file is already .mka or .mkv, set it aside
    if filename.endswith(('.mka', '.mkv')):
        filename = set_aside_original_file(filename)
        print(f"original file {original_filename} has been renamed to {filename}.")

    # Determine output file extension
    output_ext = '.mka' if has_only_audio_and_subtitles(streams) else '.mkv'

    # Create output filename
    output_filename = os.path.splitext(original_filename)[0] + output_ext

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

    print(f"removing original {filename}...", end="", flush=True)
    os.remove(filename)
    print("done.")

if __name__ == "__main__":
    for filename in argv[1:]:
        transcribe(filename)

