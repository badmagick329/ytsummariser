import subprocess
from enum import Enum
from pathlib import Path


class DownloadStatus(Enum):
    DOWNLOADED = "downloaded"
    EXISTS = "exists"
    ERROR = "error"


class DownloadedVideo:
    _output_file: str
    _video_id: str
    _yt_link: str | None = None

    def __init__(self, yt_link: str, target_dir: Path) -> None:
        self._yt_link = yt_link
        self._target_dir = target_dir
        if "?v=" in yt_link:
            self._video_id = yt_link.split("&")[0].split("?v=")[-1]
        elif "youtu.be/" in yt_link:
            self._video_id = yt_link.split("?")[0].split("youtu.be/")[-1]
        self._output_file = str(self._target_dir / f"{self._video_id}.mp3")

    def is_downloaded(self):
        return Path(self._output_file).is_file()

    def download(self) -> DownloadStatus:
        if self.is_downloaded():
            return DownloadStatus.EXISTS
        cmd = self.create_cmd()
        subprocess.run(cmd)
        return (
            DownloadStatus.DOWNLOADED if self.is_downloaded() else DownloadStatus.ERROR
        )

    def create_cmd(self) -> str:
        return f"yt-dlp -q -x --audio-format mp3 {self._yt_link} -o {self._output_file}"

    @property
    def output_file(self) -> Path:
        return Path(self._output_file)

    @property
    def video_id(self) -> str:
        return self._video_id
