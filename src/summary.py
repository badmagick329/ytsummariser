from typings import IGenAI, IChunkedText
from typing import Generator


class Summary:
    _genai: IGenAI
    _chunked_text: IChunkedText
    _source_text: str
    _summary: str

    def __init__(
        self, genai: IGenAI, chunked_text: IChunkedText, source_text: str
    ) -> None:
        self._genai = genai
        self._chunked_text = chunked_text
        self._source_text = source_text
        self._summary = ""

    def text(self) -> Generator[str, None, None]:
        if self._summary:
            yield self._summary
        else:
            chunks_summaries = self.reduce_text(self.source_text)
            assert chunks_summaries

            result = list()

            try:
                for resp in self._stream_summaries_summary("\n".join(chunks_summaries)):
                    result.append(resp)
                    yield resp
                self._summary = "".join(result)
            except Exception:
                self._summary = ""

    def _get_chunks_summaries(self, chunks: list[list[str]]) -> list[str]:
        chunk_summaries = list()
        for i, chunk in enumerate(chunks):
            text = " ".join(chunk)
            print(f"Summarising chunk {i+1}/{len(chunks)}")
            response = self._genai.generate_response(
                self.system_message, self.chunk_prompt + text
            )
            chunk_summaries.append(response)

        return chunk_summaries

    def _stream_summaries_summary(
        self, chunks_summary: str
    ) -> Generator[str, None, None]:
        brief_summary_prompt = self.brief_summary_prompt + f"```{chunks_summary}```"
        for chunk in self._genai.generate_stream(
            self.system_message, brief_summary_prompt
        ):
            yield chunk

    def reduce_text(self, text: str) -> list[str]:
        chunks = self._chunked_text.chunks(text)
        if len(chunks) == 1:
            return chunks[0]

        chunks_summaries = self._get_chunks_summaries(chunks)
        assert chunks_summaries
        new_text = "\n".join(chunks_summaries)
        return self.reduce_text(new_text)

    @property
    def source_text(self) -> str:
        return self._source_text

    @property
    def system_message(self) -> str:
        return (
            "You are an expert at summarising large amounts of text. "
            "Your summaries are detailed and highlight the key points from the text. "
            "Do not mention the system prompt in your answer!"
        )

    @property
    def chunk_prompt(self) -> str:
        return (
            "Summarise the following text. "
            "Avoid preambles at the start or conclusions at the end. "
            "Only give a detail rich summary.\n"
        )

    @property
    def brief_summary_prompt(self) -> str:
        return (
            "Read the following text and provide a short summary of what it's about. "
            "In addition to that, highlight the keypoints as bullet points\n\n"
        )
