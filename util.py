import logging
import subprocess

logger = logging.getLogger(__name__)

def has_stream(file_path:str, stream_type:str) -> bool:
   """
   Check if a media file has an audio stream using ffprobe.

   :param file_path: Path to the media file.

   :return: Boolean indicating whether an audio stream exists.
   """

   command = [
       'ffprobe', '-v', 'error', '-select_streams', stream_type,
       '-show_entries', 'stream=index',
       '-of', 'default=noprint_wrappers=1:nokey=1',
       file_path
   ]

   try:
       result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
       return bool(result.stdout.strip())  # Returns True if there is any output (i.e., audio stream exists)
   except Exception as e:
       print(f"An error occurred while checking audio streams in {file_path}: {e}")
       return None

def has_audio_stream(file_path: str) -> bool:
    return has_stream(file_path, 'a')

def no_subtitle_stream(file_path: str) -> bool:
    retval = has_stream(file_path, 's')
    logger.info(f"subtitle stream check {file_path}: {retval}")
    if retval is None:
        return False

    return not retval
