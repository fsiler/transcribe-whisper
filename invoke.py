#!/usr/bin/env python
import os
import signal
import sys
from functools import partial
from transcribe import transcribe

# Global state for signal handling
stop_after_current = False
immediate_stop = False

def signal_handler(signum, frame, confirm_exit=False):
    """
    Handles SIGINT signals.
    - First SIGINT: Stops after the current file.
    - Second SIGINT: Stops immediately.
    """
    global stop_after_current, immediate_stop

    if confirm_exit:
        print("\nSecond SIGINT received: Stopping immediately.")
        immediate_stop = True
    else:
        print("\nFirst SIGINT received: Will stop after the current file.")
        stop_after_current = True
        # Update the signal handler to require confirmation for immediate exit
        signal.signal(signal.SIGINT, partial(signal_handler, confirm_exit=True))

def get_all_files(path="~/Movies"):
    """
    Generator that yields all file paths in the ~/Movies directory.
    """
    movies_dir = os.path.expanduser(path)
    for root, _, files in os.walk(movies_dir):
        for file in files:
            yield os.path.join(root, file)

def filter_files_by_keywords(files, keywords):
    """
    Generator that filters files based on case-insensitive keyword matches in the filename.
    :param files: Iterable of file paths.
    :param keywords: List of keywords to search for in filenames (case-insensitive).
    """
    keywords_lower = [keyword.lower() for keyword in keywords]
    for file_path in files:
        filename_lower = os.path.basename(file_path).lower()
        if any(keyword in filename_lower for keyword in keywords_lower):
            yield file_path

def sort_files_by_size(files):
    """
    Generator that sorts files by size (smallest to largest).
    :param files: Iterable of file paths.
    """
    # Get the file paths and their sizes
    files_with_sizes = ((file_path, os.path.getsize(file_path)) for file_path in files)

    # Sort by size
    sorted_files = sorted(files_with_sizes, key=lambda x: x[1])

    # Yield only the file paths in sorted order
    for file_path, _ in sorted_files:
        yield file_path

def load_keywords_from_file(file_path="keywords.txt"):
    """
    Load keywords from a text file. Each line contains one keyword.
    :param file_path: Path to the text file containing keywords.
    :return: List of keywords.
    """
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]  # Remove empty lines and whitespace

def process_files():
    """
    Main function to process files with signal handling.
    """
    global stop_after_current, immediate_stop

    # Step 1: Get all files in ~/Movies
    all_files = get_all_files()

    # Step 2: Load keywords from a text file
    keywords = load_keywords_from_file("keywords.txt")
    
    if not keywords:
        print("No keywords found in 'keywords.txt'. Exiting.")
        sys.exit(1)

    # Step 3: Filter files by loaded keywords
    filtered_files = filter_files_by_keywords(all_files, keywords)

    # Step 4: Sort filtered files by size
    sorted_filtered_files = sort_files_by_size(filtered_files)

    # Process each file
    print("Files sorted by size:")
    
    for file in sorted_filtered_files:
        if immediate_stop:
            print("\nImmediate stop triggered. Exiting.")
            sys.exit(0)

        print(f"Processing: {file}")
        transcribe(file)

        if stop_after_current:
            print("\nStopping after current file as requested.")
            break

if __name__ == "__main__":
    # Register initial signal handler for SIGINT
    signal.signal(signal.SIGINT, partial(signal_handler, confirm_exit=False))
    
    try:
        process_files()
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
