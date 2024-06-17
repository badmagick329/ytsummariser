import sys
from pathlib import Path

from chunked_text import OllamaChunkedText, OpenAIChunkedText
from models import LlamaGen, OpenAIGen
from summary import Summary

BASE_DIR = Path(__file__).parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

DATA_DIR = BASE_DIR.parent / "data"

from downloaded_video import DownloadedVideo, DownloadStatus
from transcript import Transcript


def summarise():
    if len(sys.argv) < 2:
        print("Usage: python summarise.py <youtube_link>")
        sys.exit(1)

    yt_link = sys.argv[1]
    video_text = get_video_text(yt_link)

    chunked_text = OllamaChunkedText(max_words_per_chunk=1200, overlap=4)
    llama_gen = LlamaGen()
    # chunked_text = OpenAIChunkedText(overlap=4)
    # llama_gen = OpenAIGen()
    summary = Summary(llama_gen, chunked_text, video_text)
    for word in summary.text():
        print(word, end="", flush=True)
    print()


def get_video_text(yt_link: str) -> str:
    downloaded_video = DownloadedVideo(yt_link, DATA_DIR)

    transcript = Transcript(downloaded_video.mp3_output)

    if transcript.load_saved_transcript(downloaded_video.txt_output):
        print("Saved transcript found. Loading...")
    else:
        print("Downloading file...")
        downloaded_video = download_video(downloaded_video)
        print("Generating transcript...")
        transcript.generate_new_transcript(downloaded_video.txt_output)

    return transcript.text()


def download_video(video: DownloadedVideo) -> DownloadedVideo:
    download_status = video.download()
    if download_status == DownloadStatus.ERROR:
        print(f"Error downloading the file: {video.mp3_output}")
        sys.exit(1)

    if download_status == DownloadStatus.EXISTS:
        print(f"File already exists: {video.mp3_output}")
    else:
        print(f"Downloaded file: {video.mp3_output}")
    return video


if __name__ == "__main__":
    summarise()
