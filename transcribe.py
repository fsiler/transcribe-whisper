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

def get_file_streams(src_fn):
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', src_fn]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return data['streams']

def has_subtitle_stream(streams):
    return any(stream['codec_type'] == 'subtitle' for stream in streams)

def has_only_audio_and_subtitles(streams):
    return all(stream['codec_type'] in ['audio', 'subtitle'] for stream in streams)

def set_aside_original_file(src_fn):
    temp_filename = src_fn + '.temp'
    shutil.move(src_fn, temp_filename)
    return temp_filename

def transcribe(src_fn):
    start_time = time.time()
    print(f"=== examining {src_fn}: ", end="", flush=True)
    streams = get_file_streams(src_fn)

    if has_subtitle_stream(streams):
        print(f"file has subtitle stream, skipping.")
        return

    # Determine output file extension
    output_ext = '.mka' if has_only_audio_and_subtitles(streams) else '.mkv'

    # Create temporary output filename
    dest_fn = os.path.splitext(src_fn)[0] + output_ext + '.temp'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16 = False if device=='cpu' else True

    print("loading model...", end="", flush=True)
    model = whisper.load_model("turbo").to(device)
    print("done.", flush=True)

    print(f"    opening {src_fn}....")

    # Get audio length
    audio = whisper.load_audio(src_fn)
    audio_length = len(audio) / whisper.audio.SAMPLE_RATE

    result = model.transcribe(src_fn, word_timestamps=True, fp16=fp16)

    # Prepare SRT content
    srt_content = ""
    for i, segment in enumerate(result["segments"]):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        srt_content += f"{i+1}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n\n"

    # Write transcription to temporary MKA/MKV file
    cmd = [
        'ffmpeg', '-i', src_fn,
        '-i', '-',
        '-map', '0', '-map', '1',
        '-c', 'copy',
        '-c:s', 'srt',
        dest_fn
    ]

    subprocess.run(cmd, input=srt_content.encode('utf-8'), check=True)

    # End timing the transcription
    end_time = time.time()
    transcription_time = end_time - start_time

    # Calculate ratio
    ratio = audio_length / transcription_time

    print(f"Transcription file created: {dest_fn}")
    print(f"Audio length: {format_timestamp(audio_length)}")
    print(f"Transcription time: {format_timestamp(transcription_time)}")
    print(f"Transcribed at {ratio:.2f}x speed")

    # Replace the original file with the new one
    print(f"replacing original {src_fn} with {dest_fn}...", end="", flush=True)
    os.replace(dest_fn, src_fn)
    print("done.")

if __name__ == "__main__":
    for src_fn in argv[1:]:
        transcribe(src_fn)
