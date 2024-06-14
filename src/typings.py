from typing import Generator, Protocol


class IGenAI(Protocol):
    def generate_stream(
        self, system_message: str, prompt: str
    ) -> Generator[str, None, None]: ...

    def generate_response(self, system_message: str, prompt: str) -> str: ...


class IChunkedText(Protocol):
    """
    The source text is chunked into a list of sentences. These chunks are
    broken up into a list of chunks based on the maximum context length for
    a specific model
    """

    def chunks(self, source_text: str) -> list[list[str]]:
        """
        Args:
            source_text (str): The text to be chunked
        Returns:
            list[list[str]]: A list of chunks. The chunks are a list of sentences

        """
        ...
