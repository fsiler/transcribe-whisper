#!/usr/bin/env python
import os
import signal
import sys
import subprocess
import re
from pathlib import Path  # Importing Path from pathlib

import torch
import whisper
from transcribe import transcribe

type FileMatch = (str, re.Match)

# Global state for signal handling
STOP_AFTER_CURRENT = False

def signal_handler_first(signum, frame) -> None:
    """
    Handles the first SIGINT signal.
    Sets the program to stop after processing the current file.
    """
    global STOP_AFTER_CURRENT

    print("\nFirst SIGINT received: Will stop after the current file.")
    STOP_AFTER_CURRENT = True
    # Update the signal handler to handle the second SIGINT
    signal.signal(signal.SIGINT, signal_handler_second)

def signal_handler_second(signum, frame) -> None:
    """
    Handles the second SIGINT signal.
    Stops the program immediately.
    """
    print("\nSecond SIGINT received: Stopping immediately.")
    sys.exit()

def get_all_files(path: str = "~/Movies") -> list[str]:
    """
    Generator that yields all file paths in the ~/Movies directory.
    """
    movies_dir = os.path.expanduser(path)
    for root, _, files in os.walk(movies_dir):
        for file in files:
            yield os.path.join(root, file)

def filter_files_by_keywords(files: list[str], keywords: set[str]) -> FileMatch:
    """
    Generator that filters files based on case-insensitive keyword matches in the filename.

    :param files: Iterable of file paths.
    :param keywords: Set of keywords to search for in filenames (case-insensitive).
    """

    # Create a single regex pattern from all keywords
    keywords_piped = f'({"|".join(re.escape(keyword) for keyword in sorted(keywords))})'
    keyword_pattern = rf'(\b{keywords_piped}|{keywords_piped}\b)'
    print(f"matching pattern: {keyword_pattern}")

    keyword_matcher = re.compile(keyword_pattern, re.IGNORECASE)

    for file_path in files:
        fn = os.path.basename(file_path)
        if matcher := keyword_matcher.search(fn):
            yield (file_path, matcher)

def sort_files_by_size(files: list[FileMatch]) -> list[FileMatch]:
    """
    Generator that sorts files by size (smallest to largest).

    :param files: Iterable of file paths.
    """
    # Get the file paths and their sizes
    files_with_sizes = ((file_path, os.path.getsize(file_path[0])) for file_path in files)

    # Sort by size
    sorted_files = sorted(files_with_sizes, key=lambda x: x[1])

    # Yield only the file paths in sorted order
    for file_path, _ in sorted_files:
        yield file_path

def load_keywords_from_file(file_path: str = "keywords.txt"):
    """
    Load keywords from a text file. Each line contains one keyword.

    :param file_path: Path to the text file containing keywords.

    :return: Set of keywords.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # Remove empty lines and whitespace
        return {line.strip() for line in f if line.strip() and not line.startswith('#')}

def has_stream(file_path: str, stream_type: str) -> bool:
    """
    Check if a media file has an audio stream using ffprobe.

    :param file_path: Path to the media file.

    :return: Boolean indicating whether an audio stream exists.
    """

    command = [
        'ffprobe', '-v', 'error', '-select_streams', stream_type,
        '-show_entries', 'stream=index',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return bool(result.stdout.strip())  # Returns True if there is any output (i.e., audio stream exists)
    except Exception as e:
        print(f"An error occurred while checking audio streams in {file_path}: {e}")
        return False

def has_audio_stream(file_path: str) -> bool:
    return has_stream(file_path, 'a')

def has_subtitle_stream(file_path: str) -> bool:
    return has_stream(file_path, 's')

def add_null_subtitle_stream(file_path: str):
    """
    Add a null subtitle stream to a media file using FFmpeg.

    :param file_path: Path to the media file.
    :return: None
    """
    output_file = f"{Path(file_path).stem}_with_null_subs{Path(file_path).suffix}"

    command = [
        'ffmpeg', '-i', file_path,
        '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
        '-c:v', 'copy', '-c:a', 'copy',
        '-c:s', 'mov_text',
        '-metadata:s:s:0', 'language=eng',
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Added null subtitle stream to {file_path}. Output saved as {output_file}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to add null subtitle stream to {file_path}: {e}")

def process_files(model_type:str="turbo"):
    """
    Main function to process files with signal handling.
    """
    keywords = load_keywords_from_file("keywords.txt")
    if not keywords:
        print("No keywords found in 'keywords.txt'. Exiting.")
        sys.exit(1)

    print("loading model...", end="", flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model(model_type).to(device)
    print("done.", flush=True)

    # Step 1: Get all files in ~/Movies
    all_files = get_all_files()

    # Step 3: Filter files by loaded keywords
    filtered_files = filter_files_by_keywords(all_files, keywords)

    # Step 4: Sort filtered files by size
    sorted_filtered_files = sort_files_by_size(filtered_files)

    for file, matcher in sorted_filtered_files:
        matchword = matcher[0]
        # Check if the file already has a subtitle stream
        if has_subtitle_stream(file):
            print(f"Skipping {file}: already has a subtitle stream. (matched on '{matchword}')")
            continue

        # Check if there are audio streams before checking levels
        if not has_audio_stream(file):
            print(f"Skipping {file}: no audio streams found. (matched on '{matchword}')")
            continue

#        # Check audio levels before transcribing
#        if not analyze_audio_levels(file):
#            print(f"Audio levels are insufficient for transcription.")
##            add_null_subtitle_stream(file)
#            continue

        print(f"=== found {file}, matches '{matchword}'")
        transcribe(file, model=model)

        if STOP_AFTER_CURRENT:
            print("\nStopping after current file as requested.")
            break

if __name__ == "__main__":
    # Register initial signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler_first)

    try:
        process_files()
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
