"""ffmpeg / ffprobe wrappers: metadata, audio extraction, streaming frame decode.

Frames are decoded at a low fps via an ffmpeg rawvideo pipe (no temp files).
Analysis uses grayscale (1 byte/pixel); export re-extracts full-resolution color.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(
            f"`{name}` not found on PATH. Install ffmpeg (it provides ffmpeg and ffprobe)."
        )
    return exe


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def probe(video_path: str | Path) -> VideoInfo:
    """Return basic stream metadata via ffprobe."""
    video_path = Path(video_path)
    cmd = [
        _tool("ffprobe"),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)

    v_stream = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if v_stream is None:
        raise ValueError(f"No video stream in {video_path}")
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])

    num, den = (v_stream.get("r_frame_rate", "25/1").split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 25.0
    duration = float(data["format"].get("duration", v_stream.get("duration", 0.0)))

    return VideoInfo(
        path=video_path,
        duration=duration,
        width=int(v_stream["width"]),
        height=int(v_stream["height"]),
        fps=fps,
        has_audio=has_audio,
    )


def extract_audio(
    video_path: str | Path,
    out_wav: str | Path,
    start: float | None = None,
    duration: float | None = None,
    sample_rate: int = 16000,
) -> Path:
    """Extract mono PCM WAV (default 16 kHz) suitable for Whisper."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video_path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(out_wav)]
    subprocess.run(cmd, check=True)
    return out_wav


def _even(n: int) -> int:
    return n - (n % 2)


def target_size(info: VideoInfo, width: int | None) -> tuple[int, int]:
    """Compute (w, h) for analysis, preserving aspect ratio (even dims)."""
    if width is None or width >= info.width:
        return info.width, info.height
    w = _even(width)
    h = _even(round(width * info.height / info.width))
    return w, h


def decode_frames(
    video_path: str | Path,
    fps: float,
    start: float | None = None,
    duration: float | None = None,
    size: tuple[int, int] | None = None,
    gray: bool = True,
) -> Iterator[tuple[float, "np_ndarray"]]:  # noqa: F821
    """Stream frames at `fps`, yielding (absolute_timestamp_seconds, frame).

    `size` is (w, h); if None, native resolution is used. `gray=True` yields a
    2-D uint8 array; otherwise a 3-D BGR uint8 array.
    """
    import numpy as np

    # Fail loudly on falsy-but-not-None values (False/0 would silently decode
    # nothing via `-t 0.000`).
    if duration is not None and not (isinstance(duration, (int, float))
                                     and not isinstance(duration, bool) and duration > 0):
        raise ValueError(f"duration must be a positive number or None, got {duration!r}")
    if start is not None and isinstance(start, bool):
        raise ValueError(f"start must be a number or None, got {start!r}")

    vf = [f"fps={fps}"]
    if size is not None:
        vf.append(f"scale={size[0]}:{size[1]}")
    if gray:
        vf.append("format=gray")
    pix_fmt = "gray" if gray else "bgr24"

    cmd = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video_path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-an", "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]

    # We need the output dims to size the byte reads.
    if size is None:
        info = probe(video_path)
        w, h = info.width, info.height
    else:
        w, h = size
    channels = 1 if gray else 3
    frame_bytes = w * h * channels
    shape = (h, w) if gray else (h, w, 3)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
    base = start or 0.0
    i = 0
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
            yield base + i / fps, frame
            i += 1
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait()


def extract_frame(
    video_path: str | Path,
    timestamp: float,
    out_path: str | Path,
    gray: bool = False,
) -> Path:
    """Extract a single full-resolution frame at `timestamp` to an image file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{timestamp:.3f}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "2",
    ]
    if gray:
        cmd += ["-vf", "format=gray"]
    cmd += [str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path
