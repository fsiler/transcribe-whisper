#!/usr/bin/env python
import os
import re
import signal
import asyncio

from asyncio    import PriorityQueue
from functools  import partial
from itertools  import islice
from typing     import Generator, Iterable, Tuple

import whisper

from torch.cuda import is_available as cuda_is_available
from transcribe import transcribe
from util       import has_audio_stream, has_subtitle_stream

# Global flag to indicate when to stop
stop_flag = False

def signal_handler(signum, frame):
    global stop_flag
    print("\nReceived SIGINT. Stopping after current transcription...")
    stop_flag = True

signal.signal(signal.SIGINT, signal_handler)

def get_all_files(path: str = "~/Movies") -> Generator[tuple[int,str], None, None]:
    movies_dir = os.path.expanduser(path)
    for root, _, files in os.walk(movies_dir):
        for file in files:
            file_path = os.path.join(root, file)
            yield (os.path.getsize(file_path), file_path)

def load_keywords_from_file(file_path: str = "keywords.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        keywords = {line.strip() for line in f if line.strip() and not line.startswith('#')}

    keywords_piped = f'({"|".join(re.escape(keyword) for keyword in sorted(keywords))})'
    keyword_pattern = rf'(\b{keywords_piped}|{keywords_piped}\b)'
    print(f"matching pattern: {keyword_pattern}")
    return re.compile(keyword_pattern, re.IGNORECASE)

def filter_keywords(file_path, keyword_matcher:re.Pattern) -> [str,re.Match]:
    if matcher := keyword_matcher.search(file_path):
        return (file_path, matcher)

    return None

#def filter_files(files: Iterable[tuple[int, str]], exclude_suffixes: tuple = ('.srt', '.xz')) -> Generator[tuple[int, str], None, None]:
#    keyword_matcher = load_keywords_from_file()
#    for size, path in files:
#        if not path.endswith(exclude_suffixes) and filter_keywords(path, keyword_matcher) and has_audio_stream(path) and not has_subtitle_stream(path):
#            yield (size, path)

def filter_files(files: Iterable[Tuple[int, str]], exclude_suffixes: Tuple[str, ...] = ('.srt', '.xz')) -> Generator[Tuple[int, str], None, None]:
    keyword_matcher = load_keywords_from_file()

    def is_valid_file(path: str) -> bool:
        return (
            not path.endswith(exclude_suffixes) and
            filter_keywords(path, keyword_matcher) and
            has_audio_stream(path) and
            not has_subtitle_stream(path)
        )

    yield from ((size, path) for size, path in files if is_valid_file(path))

def take(n:int, iterable):
    return list(islice(iterable, n))

async def pprint_pq(pq):
    print("Priority Queue Contents:")
    print("----------------------")
    print("Priority | Value")
    print("----------------------")
    while not pq.empty():
        priority, value = await pq.get()
        print(f"{priority:8} | {value}")
    print("----------------------")

async def main():
    q = PriorityQueue()

    print("loading model...", end="", flush=True)
    device = 'cuda' if cuda_is_available() else 'cpu'
    model = whisper.load_model("turbo").to(device)
    print("done.", flush=True)

    all_files = get_all_files()
    filtered_files = filter_files(all_files)

    while not q.empty() or True:
        if stop_flag:
            break

        # every iteration, let's get 100 candidates available, then process the smallest one
        for i in take(100, filtered_files):
            await q.put(i)

        if q.empty():
            break

        size, file_path = await q.get()
        transcribe(file_path, model=model)

        # Give a chance for the event loop to process the SIGINT
        await asyncio.sleep(0)

if __name__ == "__main__":
    asyncio.run(main())
