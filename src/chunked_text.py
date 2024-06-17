from nltk.tokenize import sent_tokenize, word_tokenize
import tiktoken

# import nltk

# nltk.download("punkt")


class OllamaChunkedTextError(Exception):
    pass


class OpenAIChunkedTextError(Exception):
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
            max_words_per_chunk (int, optional): An upper limit on the
            number of words per chunk. Once this limit is reached, more
            sentences won't be added to the chunk. Defaults to 3000.

            min_max_words_per_chunk (int, optional): A lower limit on the
            number of words per chunk. Defaults to 20.

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

    def _create_chunks(self, source_text: str):
        sentences = sent_tokenize(source_text, language=self._language)
        if not sentences:
            self._chunks = list()

        chunks = []
        chunk = []
        for sent in sentences:
            number_of_words_in_sent = len(word_tokenize(sent))
            if number_of_words_in_sent > self._max_words_per_chunk:
                raise OllamaChunkedTextError(
                    f"max_words_per_chunk is too small for the sentence: {sent}"
                )

            if not chunk:
                chunk = [sent]
                continue

            new_number_of_tokens = len(word_tokenize(" ".join(chunk) + f" {sent}"))
            if new_number_of_tokens <= self._max_words_per_chunk:
                chunk.append(sent)
                continue

            chunks.append(chunk)
            chunk = [sent]
            if self._overlap > 0 and chunks:
                last_chunk = chunks[-1]
                chunk = [last_chunk[-(self._overlap)]]

        if chunk:
            chunks.append(chunk)

        self._chunks = chunks


class OpenAIChunkedText:
    _model: str
    _overlap: int
    _language: str
    _chunks: list[list[str]] | None
    _max_context = {
        "gpt-4o": 128_000,
        "gpt-4-turbo": 128_000,
        "gpt-3.5-turbo": 16_385,
    }

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        overlap: int = 0,
        language: str = "english",
    ):
        """
        Args:
            model (str, optional): Defaults to "gpt-3.5-turbo".

            overlap (int, optional): Number of overlapping sentences in chunks. Defaults to 0.

            language (str, optional): Defaults to "english".
        """
        self._model = model
        self._overlap = overlap
        self._language = language
        self._chunks = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_tokens_per_chunk(self) -> int:
        return self._max_context[self.model]

    def chunks(self, source_text: str) -> list[list[str]]:
        if self._chunks is not None:
            return self._chunks

        self._create_chunks(source_text)
        assert self._chunks is not None
        return self._chunks

    def _create_chunks(self, source_text: str):
        sentences = sent_tokenize(source_text, language=self._language)
        if not sentences:
            self._chunks = list()

        chunks = []
        chunk = []
        for sent in sentences:
            number_of_tokens_in_sent = self._num_tokens_from_string(sent)
            if number_of_tokens_in_sent > self.max_tokens_per_chunk:
                raise OpenAIChunkedTextError(
                    f"max_tokens_per_chunk is too small for the sentence: {sent}"
                )

            if not chunk:
                chunk = [sent]
                continue

            new_number_of_tokens = self._num_tokens_from_string(
                " ".join(chunk) + f" {sent}"
            )
            if new_number_of_tokens <= self.max_tokens_per_chunk:
                chunk.append(sent)
                continue

            chunks.append(chunk)
            chunk = [sent]
            if self._overlap > 0 and chunks:
                last_chunk = chunks[-1]
                chunk = [last_chunk[-(self._overlap)]]

        if chunk:
            chunks.append(chunk)

        self._chunks = chunks

    def _num_tokens_from_string(self, text: str):
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))
