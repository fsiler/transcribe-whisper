# `whisper-transcribe`
This is meant to be a simple command-line utility.
It uses OpenAI Whisper to transcribe audio.

## Usage
Invoke it by calling `transcribe.py` with any number of media files.
Under the hood, `ffmpeg` is used to transcode, so a large number of file formats should work, including `.mp3`, `.m4a`, and `.mkv`.
Any file with an audio stream should work, including video files.
The output is a `.vtt` subtitle file in the same folder as the origin.
