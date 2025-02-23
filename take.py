#!/usr/bin/env python
import asyncio
import logging
import os
import re
import signal

from asyncio    import PriorityQueue
from functools  import partial
from itertools  import islice
from sys        import exit
from typing     import Generator, Iterable, Tuple

import whisper

from torch.cuda import is_available as cuda_is_available
from transcribe import transcribe
from util       import has_audio_stream, no_subtitle_stream

# Global flag to indicate when to stop
STOP_FLAG = 0

logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    global STOP_FLAG

    logging.error("\nReceived SIGINT. Stopping after current transcription...")
    STOP_FLAG += 1

    if STOP_FLAG > 1:
        exit()

def get_audio_length_seconds(fn: str) -> float:
    try:
        audio = whisper.load_audio(fn)
        return len(audio) / whisper.audio.SAMPLE_RATE
    except Exception as e:
        logging.debug(f"! couldn't get audio for {fn}: {e}")
        raise ValueError

def get_all_files(path: str = "~/Movies") -> Generator[str, None, None]:
    movies_dir = os.path.expanduser(path)
    for root, _, files in os.walk(movies_dir):
        for file in files:
            file_path = os.path.join(root, file)
            logging.debug(f"found file {file_path}")
            yield file_path

def load_keywords_from_file(file_path: str = "keywords.txt") -> re.Pattern:
    with open(file_path, "r", encoding="utf-8") as f:
        keywords = {line.strip() for line in f if line.strip() and not line.startswith('#')}

    keywords_piped = f'({"|".join(re.escape(keyword) for keyword in sorted(keywords))})'
    keyword_pattern = rf'(\b{keywords_piped}|{keywords_piped}\b)'
    logging.info(f"matching pattern: {keyword_pattern}")
    return re.compile(keyword_pattern, re.IGNORECASE)

def take[T](n:int, iterable:Iterable[T]) -> list[T]:
    return list(islice(iterable, n))

async def main() -> None:
    global STOP_FLAG

    q = PriorityQueue()

    all_files = get_all_files()

    filter1 = filter(lambda fn: not fn.endswith( ('srt','xz')), all_files)

    keyword_matcher = load_keywords_from_file()
#    filter2 = filter(lambda fn: keyword_matcher.search(fn), filter1)

    filter3 = filter(no_subtitle_stream, filter1)
#    filter4 = filter(has_audio_stream, filter3)

    logging.info("loading model...")
    device = 'cuda' if cuda_is_available() else 'cpu'
    model = whisper.load_model("turbo").to(device)
    logging.info("done.")

    while not q.empty() or True:
        if STOP_FLAG:
            break

        # every iteration, let's find candidates, then process the smallest one overall
        for fn in take(25, filter3):
            try:
                item = (get_audio_length_seconds(fn), fn)
            except ValueError:
                logging.info("! couldn't get audio for {fn}")
            logging.info(f"= queuing {item} =")
            await q.put(item)

        if q.empty():
            break

        _, file_path = await q.get()
        transcribe(file_path, model=model)

        # Give a chance for the event loop to process the SIGINT
        await asyncio.sleep(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
