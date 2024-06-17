from typing import Generator
import ollama
from openai import OpenAI


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


class OpenAIGen:
    _model: str
    _client: OpenAI

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        self._model = model
        self._client = OpenAI()

    @property
    def model(self) -> str:
        return self._model

    def generate_stream(
        self, system_message: str, prompt: str
    ) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def generate_response(self, system_message: str, prompt: str):
        chat_completion = self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
        )
        return chat_completion.choices[0].message.content or ""  # type:ignore
