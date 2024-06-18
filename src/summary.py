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
            chunks = self._chunked_text.chunks(self.source_text)
            if len(chunks) == 1:
                chunks_summaries = chunks[0]
            else:
                chunks_summaries = self._get_chunks_summaries(
                    self._chunked_text.chunks(self.source_text)
                )
                assert chunks_summaries

            result = list()

            try:
                for resp in self._stream_summaries_summary(chunks_summaries):
                    result.append(resp)
                    yield resp
                self._summary = "".join(result)
            except Exception:
                self._summary = ""

    @property
    def source_text(self) -> str:
        return self._source_text

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
        self, summaries: list[str]
    ) -> Generator[str, None, None]:
        chunks_summary = "\n".join(summaries)
        brief_summary_prompt = self.brief_summary_prompt + f"```{chunks_summary}```"
        for chunk in self._genai.generate_stream(
            self.system_message, brief_summary_prompt
        ):
            yield chunk

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
