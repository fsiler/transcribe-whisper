# `transcribe.py`
This is meant to be a simple command-line utility.
It uses [OpenAI Whisper](https://github.com/openai/whisper) to add text transcripts to media files containing audio.

## Installation
```yaml
python -m pip install -r requirements.txt
```

## Usage
Invoke it by calling `transcribe.py` with any number of media files.
Under the hood, `ffmpeg` is used to transcode, so a large number of file formats should work, including `.mp3`, `.webm`, `.m4a`, and `.mkv`.
Any file with an audio stream should work, including video files.
The output will be a blended `.mka` or `.mkv` containing your original streams plus subtitles.
