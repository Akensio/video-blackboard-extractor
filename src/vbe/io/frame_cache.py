"""Disk cache of low-fps grayscale analysis frames.

Frames are stored in a single memory-mapped uint8 array (N, H, W) so windowed
access over a multi-hour video never loads everything into RAM. Keyed by the
source file (name + size + mtime) plus decode parameters, so re-runs are cheap.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import ffmpeg


def _key(info: ffmpeg.VideoInfo, fps: float, size: tuple[int, int],
         start: float | None, duration: float | None) -> str:
    st = info.path.stat()
    raw = f"{info.path.name}|{st.st_size}|{int(st.st_mtime)}|{fps}|{size}|{start}|{duration}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


@dataclass
class FrameCache:
    root: Path
    times: np.ndarray          # (N,) absolute timestamps (seconds)
    frames: np.memmap          # (N, H, W) uint8 grayscale
    fps: float

    @property
    def n(self) -> int:
        return len(self.times)

    @property
    def height(self) -> int:
        return self.frames.shape[1]

    @property
    def width(self) -> int:
        return self.frames.shape[2]

    def get(self, i: int) -> np.ndarray:
        return np.asarray(self.frames[i])

    def window_indices(self, center_i: int, half: int) -> tuple[int, int]:
        return max(0, center_i - half), min(self.n, center_i + half + 1)

    def nearest_index(self, t: float) -> int:
        return int(np.argmin(np.abs(self.times - t)))

    @classmethod
    def load_or_build(
        cls,
        video_path: str | Path,
        info: ffmpeg.VideoInfo,
        fps: float,
        analysis_width: int,
        cache_root: str | Path,
        start: float | None = None,
        duration: float | None = None,
        rebuild: bool = False,
    ) -> "FrameCache":
        size = ffmpeg.target_size(info, analysis_width)
        key = _key(info, fps, size, start, duration)
        root = Path(cache_root) / key
        meta_path = root / "meta.json"
        data_path = root / "frames.dat"

        if meta_path.is_file() and data_path.is_file() and not rebuild:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            times = np.array(meta["times"], dtype=np.float64)
            h, w = meta["height"], meta["width"]
            frames = np.memmap(data_path, dtype=np.uint8, mode="r", shape=(len(times), h, w))
            return cls(root=root, times=times, frames=frames, fps=fps)

        root.mkdir(parents=True, exist_ok=True)
        w, h = size
        span = duration if duration is not None else info.duration
        n_est = int(np.ceil(span * fps)) + 8
        frames = np.memmap(data_path, dtype=np.uint8, mode="w+", shape=(n_est, h, w))

        times: list[float] = []
        total = int(np.ceil(span * fps))
        for t, frame in tqdm(
            ffmpeg.decode_frames(video_path, fps, start, duration, size=size, gray=True),
            total=total, desc="decoding frames", unit="f",
        ):
            i = len(times)
            if i >= n_est:  # safety: grow rarely needed
                break
            frames[i] = frame
            times.append(t)

        n = len(times)
        frames.flush()
        # Re-open a tight view limited to the frames we actually wrote.
        frames = np.memmap(data_path, dtype=np.uint8, mode="r", shape=(n, h, w))
        times_arr = np.array(times, dtype=np.float64)
        meta_path.write_text(
            json.dumps({"times": times, "height": h, "width": w, "fps": fps}),
            encoding="utf-8",
        )
        return cls(root=root, times=times_arr, frames=frames, fps=fps)
