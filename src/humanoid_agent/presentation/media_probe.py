"""Read typed video metadata without imposing exporter-specific validation rules."""

import subprocess
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel, Field


class VideoStreamProbe(BaseModel):
    nb_frames: str
    r_frame_rate: str
    duration: str | None = None
    width: int | None = None
    height: int | None = None

    @property
    def frame_count(self) -> int:
        return int(self.nb_frames)

    @property
    def frame_rate(self) -> Fraction:
        return Fraction(self.r_frame_rate)

    @property
    def duration_seconds(self) -> float:
        if self.duration is None:
            raise ValueError("Video stream has no recorded duration")
        return float(self.duration)


class VideoFormatProbe(BaseModel):
    duration: str


class VideoProbe(BaseModel):
    streams: list[VideoStreamProbe] = Field(min_length=1)
    format: VideoFormatProbe

    @property
    def video(self) -> VideoStreamProbe:
        return self.streams[0]


def probe_video(path: Path) -> VideoProbe:
    """Select the first video stream, keeping stream and container durations distinct."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,duration,r_frame_rate,width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return VideoProbe.model_validate_json(result.stdout)
