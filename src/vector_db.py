import time
from pathlib import Path

import chromadb
import ollama
from nltk.tokenize import sent_tokenize


class VectorDB:
    _collection: chromadb.Collection
    _embed_mode = "nomic-embed-text"

    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            db_path.mkdir(parents=True)
        assert db_path.is_dir(), "Error creating the database directory"
        self.chroma = chromadb.PersistentClient(str(db_path))

    def get_or_create_collection(
        self, collection_name: str, filename: Path
    ) -> chromadb.Collection:
        self._collection = self.chroma.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        if self._collection.count() == 0:
            self._load_collection(filename)

        return self.collection

    @property
    def collection(self) -> chromadb.Collection:
        return self._collection

    def _load_collection(self, filename: Path):
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        start_time = time.time()
        chunks = self._chunk_text_by_sentences(
            source_text=text, sentences_per_chunk=7, overlap=0
        )
        print(f"with {len(chunks)} chunks")
        for index, chunk in enumerate(chunks):
            embed = ollama.embeddings(model=self._embed_mode, prompt=chunk)["embedding"]
            self.collection.upsert(
                [str(filename) + str(index)],
                [embed],
                documents=[chunk],
                metadatas={"source": str(filename)},
            )

        print("Time taken: %s seconds" % (time.time() - start_time))

    @staticmethod
    def _chunk_text_by_sentences(
        source_text: str,
        sentences_per_chunk: int,
        overlap: int,
        language="english",
    ) -> list[str]:
        """
        Splits text by sentences
        """
        if sentences_per_chunk < 2:
            raise ValueError("The number of sentences per chunk must be 2 or more.")
        if overlap < 0 or overlap >= sentences_per_chunk - 1:
            raise ValueError(
                "Overlap must be 0 or more and less than the number of sentences per chunk."
            )

        sentences = sent_tokenize(source_text, language=language)
        if not sentences:
            print("Nothing to chunk")
            return []

        chunks = []
        i = 0
        print(len(sentences))
        while i < len(sentences):
            end = min(i + sentences_per_chunk, len(sentences))
            chunk = " ".join(sentences[i:end])

            if overlap > 0 and i > 1:
                overlap_start = max(0, i - overlap)
                overlap_end = i
                overlap_chunk = " ".join(sentences[overlap_start:overlap_end])
                chunk = overlap_chunk + " " + chunk

            chunks.append(chunk.strip())
            i += sentences_per_chunk

        return chunks
