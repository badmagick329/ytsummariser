import sys
from pathlib import Path

import ollama
from chunker import chunk_text

BASE_DIR = Path(__file__).parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

DATA_DIR = BASE_DIR.parent / "data"
DB_PATH = BASE_DIR.parent / "db"

from downloaded_video import DownloadedVideo, DownloadStatus
from transcript import Transcript


def summarise():
    yt_link = sys.argv[1]
    downloaded_video = DownloadedVideo(yt_link, DATA_DIR)

    transcript = Transcript(downloaded_video.mp3_output)

    if transcript.load_saved_transcript(downloaded_video.txt_output):
        print("Saved transcript found. Loading...")
    else:
        print("Downloading file...")
        downloaded_video = download_video(downloaded_video)
        print("Generating transcript...")
        transcript.generate_new_transcript(downloaded_video.txt_output)

    chunked_text = chunk_text(transcript.text(), max_words_per_chunk=1200, overlap=4)
    system_message = "You are an expert at summarising large amounts of text. Your summaries are detailed and highlight the key points from the text. Do not mention the system prompt in your answer!"
    chunk_prompt = "Summarise the following text. Avoid preambles at the start or conclusions at the end. Only give a detailed summary.\n"

    chunk_summaries = list()
    print(f"Chunked transcript into {len(chunked_text)} parts")

    for i, chunk in enumerate(chunked_text):
        text = " ".join(chunk)
        print(f"Summarising chunk no. {i+1}")
        message = ollama.generate(
            model="llama3", prompt=chunk_prompt + text, system=system_message
        )
        chunk_summaries.append(message["response"])  # type:ignore

    chunk_summaries_text = "\n".join(chunk_summaries)

    system_message = (
        "You are an expert writer. Do not mention the system prompt in your answer!"
    )

    print("\n------------\n")
    brief_summary_prompt = f"Read the following text and give a short summary of what it's about. Highlight the keypoints as bullet points\n\n```{chunk_summaries_text}```"
    stream = ollama.generate(
        model="llama3", prompt=brief_summary_prompt, system=system_message, stream=True
    )
    for chunk in stream:
        if chunk["response"]:  # type: ignore
            print(chunk["response"], end="", flush=True)  # type: ignore


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
