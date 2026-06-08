"""Write transcript outputs: json (segments + words), srt, vtt, txt."""
from __future__ import annotations

import json
from pathlib import Path

from .engine import TranscriptResult


def _ts(seconds: float, sep: str) -> str:
    """Format seconds as HH:MM:SS<sep>mmm (sep is ',' for srt, '.' for vtt)."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_json(result: TranscriptResult, path: str | Path) -> Path:
    path = Path(path)
    payload = {
        "language": result.language,
        "duration": result.duration,
        "device_used": result.device_used,
        "compute_type_used": result.compute_type_used,
        "segments": result.segments,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_srt(result: TranscriptResult, path: str | Path) -> Path:
    path = Path(path)
    lines = []
    for i, seg in enumerate(result.segments, start=1):
        lines.append(str(i))
        lines.append(f"{_ts(seg['start'], ',')} --> {_ts(seg['end'], ',')}")
        lines.append(seg["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_vtt(result: TranscriptResult, path: str | Path) -> Path:
    path = Path(path)
    lines = ["WEBVTT", ""]
    for seg in result.segments:
        lines.append(f"{_ts(seg['start'], '.')} --> {_ts(seg['end'], '.')}")
        lines.append(seg["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_txt(result: TranscriptResult, path: str | Path) -> Path:
    path = Path(path)
    text = "\n".join(seg["text"] for seg in result.segments)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def write_all(result: TranscriptResult, out_dir: str | Path, basename: str = "transcript") -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "json": write_json(result, out_dir / f"{basename}.json"),
        "srt": write_srt(result, out_dir / f"{basename}.srt"),
        "vtt": write_vtt(result, out_dir / f"{basename}.vtt"),
        "txt": write_txt(result, out_dir / f"{basename}.txt"),
    }
