#!/usr/bin/env python
import argparse
import json
import os
import platform
import shutil
import signal
import sys
import tempfile
import time

from collections.abc import Callable
from datetime        import timedelta
from pprint          import pprint, pformat
from subprocess      import check_call, check_output, CalledProcessError

import torch
import whisper

# Global flags
continue_processing = True
sigint_count = 0

def time_it(func:Callable) -> Callable:
    """
    A decorator that times how long a function takes to execute.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Function '{func.__name__}' executed in {format_timestamp(elapsed_time)}. Args: {args}")
        return result  # Return the result of the original function
    return wrapper

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

def format_timestamp(seconds:int) -> str:
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{td.microseconds//10_000:02d}"

def get_file_streams(filename:str) -> dict:
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', filename]
    stdout = check_output(cmd)
    data = json.loads(stdout)
    return data['streams']

def has_subtitle_stream(streams:dict) -> bool:
    return any(stream['codec_type'] == 'subtitle' for stream in streams)

def has_only_audio_and_subtitles(streams:dict) -> bool:
    return all(stream['codec_type'] in ['audio', 'subtitle'] for stream in streams)

def copy_mod_access_times(src:str, dest:str) -> None:
    # Set the access and modification times of dest to match src
    st_info = os.stat(src)
    os.utime(dest, (st_info.st_atime, st_info.st_mtime))

def is_program_available(program:str) -> bool:
    """Check if a program is available on the system."""
    return shutil.which(program) is not None

def copy_xattrs_and_tags(src:str, dest:str):
    # Only perform xattr and tag copying on macOS (Darwin)
    if platform.system() == 'Darwin':
        # List all extended attributes
        try:
            attrs = check_output(['xattr', src]).decode().split()
            for attr in attrs:
                # Read each attribute from the source file
                value = check_output(['xattr', '-px', attr, src])
                # Write the attribute to the destination file
                check_call(['xattr', '-wx', attr, value, dest])
        except CalledProcessError as e:
            print(f"Error copying xattrs from {src} to {dest}: {e}")

        # Check if the `tag` program is available before attempting to copy Finder tags.
        if is_program_available('tag'):
            try:
                tags = check_output(['tag', '-Nl', src]).decode().strip()
                if tags:
                    check_call(['tag', '-a', tags, dest])
            except CalledProcessError as e:
                print(f"Error copying Finder tags from {src} to {dest}: {e}")
        else:
            print("The `tag` program is not available. Skipping Finder tag copying.")

def transcribe(orig_fn:str, preserve_original:bool=False, model=None) -> str:
    """returns muxed filename"""
    # Check if the file exists before proceeding.
    if not os.path.exists(orig_fn):
        print(f"File not found: {orig_fn}. Skipping.")
        return None

    print(f"=== examining {orig_fn}: ", end="", flush=True)
    try:
        streams = get_file_streams(orig_fn)
    except KeyError:
        print("file doesn't have streams.")
        return orig_fn

    if has_subtitle_stream(streams):
        print("file has subtitle stream, skipping.")
        return orig_fn

    print("")
    srt_content = get_srt(orig_fn, model=model)

    output_ext = '.mka' if has_only_audio_and_subtitles(streams) else '.mkv'
    dest_fn = os.path.splitext(orig_fn)[0] + output_ext

    with tempfile.TemporaryDirectory() as temp_dir:
        working_fn = os.path.join(temp_dir, 'temp_output.mkv')

        cmd = [
            'ffmpeg',
            '-loglevel', 'error', '-stats', '-hide_banner',
            '-i', orig_fn,
            '-i', '-',
            '-map_metadata', '0',
            '-map', '0', '-map', '1',
            '-c', 'copy',
            '-c:s', 'srt',
            working_fn
        ]

        check_output(cmd, input=srt_content.encode('utf-8'))

        # Copy xattrs and tags from orig_fn to working_fn only on macOS
        copy_mod_access_times(orig_fn, working_fn)

        # Move working file to destination
        print(f"    Moving {working_fn} to {dest_fn}...", end="", flush=True)
        shutil.move(working_fn, dest_fn)
        print("done.")

    if orig_fn != dest_fn and not preserve_original:
        copy_xattrs_and_tags(orig_fn, working_fn)
        print(f"    Removing original file {orig_fn}...", end="", flush=True)
        os.remove(orig_fn)
        print("done.")

    return dest_fn

@time_it
def get_srt(fn:str, model_type:str = "turbo", model = None) -> str:

    if not model:
        print("loading model...", end="", flush=True)
        model = whisper.load_model(model_type).to(device)
        print("done.", flush=True)

    print(f"    opening {fn}", flush=True)

    fp16 = False if device == 'cpu' else True
    result = model.transcribe(fn, word_timestamps=True, fp16=fp16)

    srt_content = ""
    for i, segment in enumerate(result["segments"]):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        srt_content += f"{i+1}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n\n"

    return srt_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files and optionally preserve the original files.")

    parser.add_argument('files', type=str, metavar='file', nargs='+',
                        help='Files to process')

    parser.add_argument('-k', '--preserve-original', action='store_true',
                        help='Preserve the original files after processing')

    args = parser.parse_args()

    # Set up signal handlers
    try:
      signal.signal(signal.SIGHUP, signal_handler)
      signal.signal(signal.SIGINT, signal_handler)
    except AttributeError as e:
      print(f"couldn't register signal handler, {e}")

    print("loading model...", end="", flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model("turbo").to(device)
    print("done.", flush=True)

    for orig_fn in args.files:
        if not continue_processing:
            print("Stopping due to received signal.")
            break

        new_fn = transcribe(orig_fn, args.preserve_original, model=model)
        # Reset SIGINT count after each file
        sigint_count = 0

    print("Processing complete.")
