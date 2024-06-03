import sys
from pathlib import Path

import ollama

from vector_db import VectorDB

BASE_DIR = Path(__file__).parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

DATA_DIR = BASE_DIR.parent / "data"
DB_PATH = BASE_DIR.parent / "db"

from downloaded_video import DownloadedVideo, DownloadStatus
from transcription import Transcription


def embedding_summary():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <youtube_link>")
        sys.exit(1)

    yt_link = sys.argv[1]

    downloaded_video = get_downloaded_video(yt_link)

    print("Getting transcription...")
    transcription = Transcription(downloaded_video.output_file)
    text_output_file = DATA_DIR / f"{downloaded_video.video_id}.txt"
    transcription.get_or_create(text_output_file)

    vector_db = VectorDB(DB_PATH)
    vector_db.get_or_create_collection(downloaded_video.video_id, text_output_file)


def summarise():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <youtube_link>")
        sys.exit(1)

    yt_link = sys.argv[1]
    downloaded_video = get_downloaded_video(yt_link)

    print("Getting transcription...")
    transcription = Transcription(downloaded_video.output_file)
    text_output_file = DATA_DIR / f"{downloaded_video.video_id}.txt"
    transcription.get_or_create(text_output_file)

    brief_query = f"Give me a summary of this transcript from a youtube video. Do it in the form of bullet points:\n\n{transcription.text()}"
    brief_system_message = "You are an expert at summarising large amounts of text. You keep things concise. Do not mention the system prompt in your answer!"
    # print("Generating brief summary...")
    # stream = ollama.generate(
    #     model="llama3", prompt=brief_query, system=brief_system_message, stream=True
    # )
    # for chunk in stream:
    #     if chunk["response"]:  # type: ignore
    #         print(chunk["response"], end="", flush=True)  # type: ignore

    # print("\n-------------------")
    print("Generating longer summary...")

    model_query = f"Give me a summary of this transcript from a youtube video:\n\n{transcription.text()}"
    system_message = "You are an expert at summarising large amounts of text. Your summaries are detailed and highlight the key points from the text. Do not mention the system prompt in your answer!"
    stream = ollama.generate(
        model="llama3", prompt=model_query, system=system_message, stream=True
    )
    for chunk in stream:
        if chunk["response"]:  # type: ignore
            print(chunk["response"], end="", flush=True)  # type: ignore
    print("\n-------------------")


def query_main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <youtube_link>")
        sys.exit(1)

    query = input("Enter your query: ").strip()

    yt_link = sys.argv[1]
    downloaded_video = get_downloaded_video(yt_link)

    print("Getting transcription...")
    transcription = Transcription(downloaded_video.output_file)
    text_output_file = DATA_DIR / f"{downloaded_video.video_id}.txt"
    transcription.get_or_create(text_output_file)

    print("Creating collection...")
    vector_db = VectorDB(DB_PATH)
    vector_db.get_or_create_collection(downloaded_video.video_id, text_output_file)
    embed_mode = "nomic-embed-text"

    while query.lower() != "q":
        print("Finding relevant docs...")
        query_embed = ollama.embeddings(model=embed_mode, prompt=query)["embedding"]
        relevant_docs = vector_db.collection.query(
            query_embeddings=[query_embed], n_results=10
        )["documents"]
        if relevant_docs is None:
            print("No relevant docs for this query")
            sys.exit(1)

        print("\n\nPRINTING EMBED\n")
        relevant_docs = relevant_docs[0]
        relevant_docs = "\n\n".join(relevant_docs)  # type: ignore
        print(relevant_docs)
        print()

        model_query = f"{query} - Answer that question using the following text as a resource: {relevant_docs}"
        stream = ollama.generate(model="llama3", prompt=model_query, stream=True)
        for chunk in stream:
            if chunk["response"]:  # type: ignore
                print(chunk["response"], end="", flush=True)  # type: ignore

        query = input("\n\nEnter next query or q to quit: ").strip()


def get_downloaded_video(yt_link: str) -> DownloadedVideo:
    downloaded_video = DownloadedVideo(yt_link, DATA_DIR)
    download_status = downloaded_video.download()
    if download_status == DownloadStatus.ERROR:
        print(f"Error downloading the file: {downloaded_video.output_file}")
        sys.exit(1)

    if download_status == DownloadStatus.EXISTS:
        print(f"File already exists: {downloaded_video.output_file}")
    else:
        print(f"Downloaded file: {downloaded_video.output_file}")
    return downloaded_video


if __name__ == "__main__":
    summarise()
