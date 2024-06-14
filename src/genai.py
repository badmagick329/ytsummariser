from typing import Generator
import ollama


class LlamaGen:
    _model: str

    def __init__(self, model: str = "llama3") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def generate_stream(
        self, system_message: str, prompt: str
    ) -> Generator[str, None, None]:
        stream = ollama.generate(
            model=self.model, prompt=prompt, system=system_message, stream=True
        )
        for chunk in stream:
            if chunk["response"]:  # type: ignore
                yield chunk["response"]  # type: ignore

    def generate_response(self, system_message: str, prompt: str) -> str:
        message = ollama.generate(
            model=self.model, prompt=prompt, system=system_message
        )
        return message["response"]  # type: ignore
