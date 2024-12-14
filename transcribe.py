#!/usr/bin/env python
import os
import time
import torch
import whisper
import subprocess
import json
import shutil
import signal
import sys
import atexit
import tempfile

from datetime import timedelta

# Global flags
continue_processing = True
sigint_count = 0

def signal_handler(signum, frame):
    global continue_processing, sigint_count

    if signum == signal.SIGHUP:
        print("\nReceived SIGHUP. Will stop after current file.")
        continue_processing = False
    elif signum == signal.SIGINT:
        sigint_count += 1
        if sigint_count == 1:
            print("\nReceived first SIGINT. Will stop after current file.")
            continue_processing = False
        else:
            print("\nReceived second SIGINT. Exiting immediately.")
            sys.exit(0)

def cleanup(working_fn):
    if os.path.exists(working_fn):
        print(f"\nCleaning up temporary file: {working_fn}")
        os.remove(working_fn)

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

def transcribe(orig_fn):
    transcription_start_time = time.time()
    print(f"=== examining {orig_fn}: ", end="", flush=True)
    streams = get_file_streams(orig_fn)

    if has_subtitle_stream(streams):
        print(f"file has subtitle stream, skipping.")
        return

    # Determine output file extension
    output_ext = '.mka' if has_only_audio_and_subtitles(streams) else '.mkv'

    # Create destination filename
    dest_fn = os.path.splitext(orig_fn)[0] + output_ext

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the working filename within the temporary directory
        working_fn = os.path.join(temp_dir, 'temp_output.mkv')

        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            fp16 = False if device=='cpu' else True

            print("loading model...", end="", flush=True)
            model = whisper.load_model("turbo").to(device)
            print("done.", flush=True)

            print(f"    opening {orig_fn}", flush=True)

            # Get audio length
            audio = whisper.load_audio(orig_fn)
            audio_length = len(audio) / whisper.audio.SAMPLE_RATE
            print(f"    found audio length {format_timestamp(audio_length)}", flush=True)

            result = model.transcribe(orig_fn, word_timestamps=True, fp16=fp16)

            # Prepare SRT content
            srt_content = ""
            for i, segment in enumerate(result["segments"]):
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment['end'])
                srt_content += f"{i+1}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n\n"

            # Write transcription to working file
            cmd = [
                'ffmpeg', '-i', orig_fn,
                '-i', '-',
                '-map', '0', '-map', '1',
                '-c', 'copy',
                '-c:s', 'srt',
                working_fn
            ]

            subprocess.run(cmd, input=srt_content.encode('utf-8'), check=True)


            # Move working file to destination
            print(f"Moving {working_fn} to {dest_fn}...", end="", flush=True)
            shutil.move(working_fn, dest_fn)
            print("done.")

            # Clean up original file if different from destination
            if orig_fn != dest_fn:
                print(f"Removing original file {orig_fn}...", end="", flush=True)
                os.remove(orig_fn)
                print("done.")

            # End timing the transcription
            transcription_end_time = time.time()
            transcription_time = transcription_end_time - transcription_start_time

            # Calculate ratio
            ratio = audio_length / transcription_time

            print(f"Output: {dest_fn}")
            print(f"Length: {format_timestamp(audio_length)}, Transcription time: {format_timestamp(transcription_time)} ({ratio:.2f}x speed)")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGHUP, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    for orig_fn in sys.argv[1:]:
        if not continue_processing:
            print("Stopping due to received signal.")
            break
        transcribe(orig_fn)
        # Reset SIGINT count after each file
        sigint_count = 0

    print("Processing complete.")
