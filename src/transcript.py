from pathlib import Path

import whisper


class Transcript:
    _text: str | None = None
    _target_file: Path
    _model: whisper.Whisper
    _result: dict[str, str | list]

    def __init__(self, target_file: Path, model_name: str = "base.en") -> None:
        self._model = whisper.load_model(model_name)
        self._target_file = target_file

    def text(self) -> str:
        if self._text is not None:
            return self._text
        self._result = self._model.transcribe(str(self._target_file))
        assert isinstance(
            self._result["text"], str
        ), f"Invalid result type. {type(self._result['text'])}"
        self._text = self._result["text"]
        return self._text

    def load_saved_transcript(self, output_file: Path) -> bool:
        if not output_file.exists():
            return False

        with open(output_file, "r", encoding="utf-8") as f:
            self._text = f.read()
        return True

    def generate_new_transcript(self, output_file: Path):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(self.text())
