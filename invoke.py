#!/usr/bin/env python
import os
import signal
import sys
import asyncio
import re
from pathlib import Path
import concurrent.futures

import torch
import whisper
from transcribe import transcribe

# Global state for signal handling
STOP_AFTER_CURRENT = False

def signal_handler_first(signum, frame) -> None:
    global STOP_AFTER_CURRENT
    print("\nFirst SIGINT received: Will stop after the current file.")
    STOP_AFTER_CURRENT = True
    signal.signal(signal.SIGINT, signal_handler_second)

def signal_handler_second(signum, frame) -> None:
    print("\nSecond SIGINT received: Stopping immediately.")
    sys.exit()

def get_all_files(path: str = "~/Movies") -> list[str]:
    movies_dir = os.path.expanduser(path)
    return [os.path.join(root, file) for root, _, files in os.walk(movies_dir) for file in files]

def filter_files_by_keywords(files: list[str], keywords: set[str]):
    keywords_piped = f'({"|".join(re.escape(keyword) for keyword in sorted(keywords))})'
    keyword_pattern = rf'(\b{keywords_piped}|{keywords_piped}\b)'
    print(f"matching pattern: {keyword_pattern}")
    keyword_matcher = re.compile(keyword_pattern, re.IGNORECASE)
    return [(file_path, matcher) for file_path in files if (matcher := keyword_matcher.search(file_path))]

def sort_files_by_size(files: list[tuple[str, re.Match]]) -> list[tuple[str, re.Match]]:
    return sorted(files, key=lambda x: os.path.getsize(x[0]))

def load_keywords_from_file(file_path: str = "keywords.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith('#')}

async def has_stream(file_path: str, stream_type: str) -> bool:
    command = [
        'ffprobe', '-v', 'error', '-select_streams', stream_type,
        '-show_entries', 'stream=index',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        return bool(stdout.strip())
    except Exception as e:
        print(f"An error occurred while checking {stream_type} streams in {file_path}: {e}")
        return False

async def has_audio_stream(file_path: str) -> bool:
    return await has_stream(file_path, 'a')

async def has_subtitle_stream(file_path: str) -> bool:
    return await has_stream(file_path, 's')

async def check_file(file: tuple[str, re.Match]):
    file_path, matcher = file
    matchword = matcher[0]
    print(f"checking {file_path}, match  {matchword}")

    if await has_subtitle_stream(file_path):
        return None  # Skip files with existing subtitle streams

    if not await has_audio_stream(file_path):
        return None  # Skip files without audio streams

    print(f">>> found file {file}")
    return file

async def enumerate_target_files(sorted_filtered_files):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # Limit to 10 threads
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, asyncio.run, check_file(file)) for file in sorted_filtered_files]
        results = await asyncio.gather(*tasks)
    return [file for file in results if file is not None]

def process_files(model_type:str="turbo"):
    keywords = load_keywords_from_file("keywords.txt")
    if not keywords:
        print("No keywords found in 'keywords.txt'. Exiting.")
        sys.exit(1)

    print("loading model...", end="", flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model(model_type).to(device)
    print("done.", flush=True)

    all_files = get_all_files()
    filtered_files = filter_files_by_keywords(all_files, keywords)
    sorted_filtered_files = sort_files_by_size(filtered_files)

    # Enumerate target files in parallel (limited to 10 threads)
    target_files = asyncio.run(enumerate_target_files(sorted_filtered_files))

    for file, matcher in target_files:
        matchword = matcher[0]
        print(f">>> found {file}, matches '{matchword}'")
        transcribe(file, model=model)

        if STOP_AFTER_CURRENT:
            print("\nStopping after current file as requested.")
            break

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler_first)
    try:
        process_files()
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
