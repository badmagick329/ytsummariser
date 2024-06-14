from nltk.tokenize import sent_tokenize, word_tokenize

# import nltk

# nltk.download("punkt")


class OllamaChunkedTextError(Exception):
    pass


class OllamaChunkedText:
    _max_words_per_chunk: int
    _overlap: int
    _language: str
    _min_max_words_per_chunk: int = 20
    _chunks: list[list[str]] | None

    def __init__(
        self,
        max_words_per_chunk: int = 3000,
        min_max_words_per_chunk: int = 20,
        overlap: int = 0,
        language: str = "english",
    ):
        """

        Args:
            max_words_per_chunk (int, optional): A loose upper limit on the
            number of words per chunk. Once this limit is reached, more
            sentences won't be added to the chunk. Defaults to 3000.

            overlap (int, optional): Number of overlapping sentences in chunks.
            Defaults to 0.

            language (str, optional): Defaults to "english".
        """
        if max_words_per_chunk < min_max_words_per_chunk:
            raise OllamaChunkedTextError(
                f"max_words_per_chunk must be at least {min_max_words_per_chunk} words long."
            )
        self._max_words_per_chunk = max_words_per_chunk
        self._overlap = overlap
        self._language = language
        self._chunks = None

    def chunks(self, source_text: str) -> list[list[str]]:
        if self._chunks is not None:
            return self._chunks

        self._create_chunks(source_text)
        assert self._chunks is not None
        return self._chunks

    def _create_chunks(self, source_text):
        sentences = sent_tokenize(source_text, language=self._language)
        if not sentences:
            self._chunks = list()

        chunks = []
        chunk = []
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            words_in_sent = word_tokenize(sent)
            if len(words_in_sent) > self._max_words_per_chunk:
                raise OllamaChunkedTextError(
                    f"max_words_per_chunk is too small for the sentence: {sent}"
                )

            word_tokens_so_far = word_tokenize(" ".join(chunk))
            if len(word_tokens_so_far) > self._max_words_per_chunk:
                chunks.append(chunk)
                chunk = []
                if self._overlap > 0 and chunks:
                    last_chunk = chunks[-1]
                    chunk = [last_chunk[-(self._overlap)]]

            if chunk:
                chunk.append(sent)
            else:
                chunk = [sent]

            i += 1
        if chunk:
            chunks.append(chunk)

        self._chunks = chunks
