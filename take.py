#!/usr/bin/env python
import os

from asyncio   import PriorityQueue, run
from itertools import islice
from pprint    import pprint
from typing    import Generator

def get_all_files(path: str = "~/Movies") -> Generator[tuple[int,str], None, None]:
    movies_dir = os.path.expanduser(path)
    for root, _, files in os.walk(movies_dir):
        for file in files:
            file_path = os.path.join(root, file)
            yield (os.path.getsize(file_path), file_path)

def take(n:int, iterable):
    return list(islice(iterable, n))

async def pprint_pq(pq):
    # Create a copy of the queue items
    items = []


    # Print items
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
    for i in take(5, get_all_files()):
        await q.put(i)

    await pprint_pq(q)

run(main())
