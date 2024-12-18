#!/usr/bin/env python
import os
import signal
import sys
from transcribe import transcribe

# Global state for signal handling
STOP_AFTER_CURRENT = False

def signal_handler_first(signum, frame):
    """
    Handles the first SIGINT signal.
    Sets the program to stop after processing the current file.
    """
    global STOP_AFTER_CURRENT

    print("\nFirst SIGINT received: Will stop after the current file.")
    STOP_AFTER_CURRENT = True
    # Update the signal handler to handle the second SIGINT
    signal.signal(signal.SIGINT, signal_handler_second)

def signal_handler_second(signum, frame):
    """
    Handles the second SIGINT signal.
    Stops the program immediately.
    """
    print("\nSecond SIGINT received: Stopping immediately.")
    sys.exit()

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
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]  # Remove empty lines and whitespace

def process_files():
    """
    Main function to process files with signal handling.
    """
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

    for file in sorted_filtered_files:
        transcribe(file)

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
