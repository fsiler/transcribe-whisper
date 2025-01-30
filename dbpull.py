#!/usr/bin/env python
import multiprocessing
import subprocess
import sqlite3
import os
import time
from collections import deque

def transcribe_video(filename, length):
    conn = sqlite3.connect('video_database.db')
    cursor = conn.cursor()

    file_root, _ = os.path.splitext(filename)
    output_file = f"{file_root}.srt"

    cmd = ['./transcribe.py', filename]
    print(f"Starting: {filename}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    print(f"Completed: {filename}")

    cursor.execute("""
    UPDATE videos
    SET has_subtitles = 1
    WHERE filename = ?
    """, (filename,))
    conn.commit()
    conn.close()

def read_keywords(filename):
    keywords = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                keywords.append(line)
    return keywords

def get_eligible_videos(limit=100, keywords_file='keywords.txt'):
    keywords = read_keywords(keywords_file)

    conn = sqlite3.connect('video_database.db')
    cursor = conn.cursor()

    # Build the OR clauses dynamically
    or_clauses = ' OR '.join([f"filename LIKE '%{keyword}%'" for keyword in keywords])

    query = f"""
    SELECT filename, length_seconds
    FROM videos
    WHERE
      has_audio = 1
      AND has_subtitles = 0
      AND (
           {or_clauses}
      )
    ORDER BY length_seconds ASC
    LIMIT ?
    """

    cursor.execute(query, (limit,))
    videos = cursor.fetchall()
    conn.close()
    return videos

def transcribe_videos(max_processes):
    pool = multiprocessing.Pool(processes=max_processes)
    active_processes = 0
    video_queue = deque()

    while True:
        if not video_queue:
            video_queue.extend(get_eligible_videos(100))
            if not video_queue:
                break  # No more videos to process

        while active_processes < max_processes and video_queue:
            video = video_queue.popleft()
            filename, length = video
            pool.apply_async(transcribe_video, args=(filename, length))
            active_processes += 1

        # Wait for a short time before checking for completed processes
        time.sleep(1)

        # Update the number of active processes
        active_processes = len(pool._cache)

    pool.close()
    pool.join()

if __name__ == "__main__":
#    max_processes = multiprocessing.cpu_count()  # Or set to a specific number
    transcribe_videos(2)
