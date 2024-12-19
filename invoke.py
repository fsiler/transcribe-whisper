#!/usr/bin/env python
import os
from transcribe import transcribe

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

# Example usage
if __name__ == "__main__":
    # Step 1: Get all files in ~/Movies
    all_files = get_all_files()

    # Step 2: Filter files by multiple keywords (e.g., "action", "comedy")
    filtered_files = filter_files_by_keywords(all_files, ["rohn", "huberman","buffett","phil town","value","brian tracy","napoleon"])

    # Step 3: Sort filtered files by size
    sorted_filtered_files = sort_files_by_size(filtered_files)

    # Output the results
    print("Files sorted by size:")
    for file in sorted_filtered_files:
        transcribe(file)

