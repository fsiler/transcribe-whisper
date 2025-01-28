#!/usr/bin/env python
import multiprocessing
import subprocess
import sqlite3
import os

def transcribe_batch(videos):
    conn = sqlite3.connect('video_database.db')
    cursor = conn.cursor()

    processes = []
    for filename, length in videos:
        file_root, _ = os.path.splitext(filename)
        output_file = f"{file_root}.srt"

#        cmd = [
#            'whisper',
#            filename,
#            '--model', 'turbo',
#            '--output_format', 'srt',
#            '--output_dir', os.path.dirname(filename)
#        ]

        cmd = ['./transcribe.py', filename]
        print(f"Starting: {filename}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((process, filename))

    # Wait for all processes in the batch to complete
    for process, filename in processes:
        process.wait()
        print(f"Completed: {filename}")
        cursor.execute("""
        UPDATE videos
        SET has_subtitles = 1
        WHERE filename = ?
        """, (filename,))
        conn.commit()

    conn.close()

def get_eligible_videos():
    conn = sqlite3.connect('video_database.db')
#    conn.enable_load_extension(True)
#    conn.load_extension('/Users/fms/build/sqlite-pcre/pcre.so')
    cursor = conn.cursor()

    query = """
    SELECT filename, length_seconds
    FROM videos
    WHERE
      has_audio = 1
      AND has_subtitles = 0
      AND (
           filename LIKE '%rohn%'
        OR filename LIKE '%demarco%'
        OR filename LIKE '%koe%'
        OR filename LIKE '%napoleon hill%'
        OR filename LIKE '%brian tracy%'
        OR filename LIKE '%degrasse%'
        )
    ORDER BY length_seconds ASC
    LIMIT 2
    """

    query = """
    SELECT filename, length_seconds
    FROM videos
    WHERE
      has_audio = 1
      AND has_subtitles = 0
    ORDER BY length_seconds ASC
    LIMIT 2
    """

#    query = """
#    SELECT filename, length_seconds
#    FROM videos
#    WHERE
#      has_audio = 1
#      AND has_subtitles = 0
#      AND filename REGEXP '(\\b(ali\\ abdaal|brian\\ tracy|chris\\ voss|dan\\ koe|degrasse|demarco|hormozi|hpcalc|huberman|jordan\\ peterson|media\\.ccc|napoleon\\ hill|noah\\ kagan|productiv|prosper|rohn|sam\\ ovens|ziglar)|(ali\\ abdaal|brian\\ tracy|chris\\ voss|dan\\ koe|degrasse|demarco|hormozi|hpcalc|huberman|jordan\\ peterson|media\\.ccc|napoleon\\ hill|noah\\ kagan|productiv|prosper|rohn|sam\\ ovens|ziglar)\\b)'
#    ORDER BY length_seconds ASC
#    LIMIT 2
#    """

    cursor.execute(query)
    eligible_videos = cursor.fetchall()
    conn.close()
    return eligible_videos

def transcribe_videos():
    while eligible_videos := get_eligible_videos():

        # Split videos into two batches
        batch_size = len(eligible_videos) // 2
        batch1 = eligible_videos[:batch_size]
        batch2 = eligible_videos[batch_size:]

        # Create two processes
        p1 = multiprocessing.Process(target=transcribe_batch, args=(batch1,))
        p2 = multiprocessing.Process(target=transcribe_batch, args=(batch2,))

        # Start both processes
        p1.start()
        p2.start()

        # Wait for both processes to complete
        p1.join()
        p2.join()

if __name__ == "__main__":
    transcribe_videos()

