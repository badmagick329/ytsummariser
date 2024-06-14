import sys
from pathlib import Path
from typing import Generator

import ollama
from chunker import chunk_text, OllamaChunkedText

BASE_DIR = Path(__file__).parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

DATA_DIR = BASE_DIR.parent / "data"
DB_PATH = BASE_DIR.parent / "db"

from downloaded_video import DownloadedVideo, DownloadStatus
from transcript import Transcript


def summarise():
    if len(sys.argv) < 2:
        print("Usage: python summarise.py <youtube_link>")
        sys.exit(1)

    yt_link = sys.argv[1]
    video_text = get_video_text(yt_link)

    chunked_text = OllamaChunkedText(
        source_text=video_text, max_words_per_chunk=1200, overlap=4
    )

    chunks_summaries = get_chunks_summaries(chunked_text.chunks())
    assert chunks_summaries

    for resp in stream_summaries_summary(chunks_summaries):
        print(resp, end="", flush=True)
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


def get_chunks_summaries(chunks: list[list[str]]) -> list[str]:
    system_message = "You are an expert at summarising large amounts of text. Your summaries are detailed and highlight the key points from the text. Do not mention the system prompt in your answer!"
    chunk_prompt = "Summarise the following text. Avoid preambles at the start or conclusions at the end. Only give a detailed summary.\n"

    chunk_summaries = list()
    print(f"Chunked transcript into {len(chunks)} parts")

    for i, chunk in enumerate(chunks):
        text = " ".join(chunk)
        print(f"Summarising chunk no. {i+1}")
        message = ollama.generate(
            model="llama3", prompt=chunk_prompt + text, system=system_message
        )
        chunk_summaries.append(message["response"])  # type:ignore

    return chunk_summaries


def stream_summaries_summary(summaries: list[str]) -> Generator[str, None, None]:
    chunks_summary = "\n".join(summaries)
    system_message = "You are an expert at summarising large amounts of text. Your summaries are detailed and highlight the key points from the text. Do not mention the system prompt in your answer!"
    brief_summary_prompt = f"Read the following text and give a short summary of what it's about. Highlight the keypoints as bullet points\n\n```{chunks_summary}```"
    stream = ollama.generate(
        model="llama3", prompt=brief_summary_prompt, system=system_message, stream=True
    )
    for chunk in stream:
        if chunk["response"]:  # type: ignore
            yield chunk["response"]  # type: ignore


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
