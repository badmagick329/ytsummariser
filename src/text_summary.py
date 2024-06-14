from chunker import chunk_text


class TextSummary:
    _original_text: str
    _summary: str

    def __init__(self, original: str):
        self._original_text = original

    @property
    def original_text(self) -> str:
        return self._original_text

    @property
    def summary(self) -> str:
        return self._summary
