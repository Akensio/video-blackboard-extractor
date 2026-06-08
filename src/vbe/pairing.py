"""Pair keyframes with the transcript.

Each keyframe is captured just before its board is erased, so the speech that
explains it runs from the *previous* keyframe up to (and slightly past) this one.
We attach that text window to each keyframe - the direct basis for notes.
"""
from __future__ import annotations

import json
from pathlib import Path

from .manifest import read_manifest, write_manifest


def _load_segments(transcript_json: str | Path) -> list[dict]:
    data = json.loads(Path(transcript_json).read_text(encoding="utf-8"))
    return data.get("segments", [])


def _text_in_window(segments: list[dict], t0: float, t1: float) -> tuple[str, float | None, float | None]:
    picked = [s for s in segments if s["end"] >= t0 and s["start"] <= t1]
    if not picked:
        return "", None, None
    text = " ".join(s["text"].strip() for s in picked).strip()
    return text, picked[0]["start"], picked[-1]["end"]


def pair_manifest(
    manifest_path: str | Path,
    transcript_json: str | Path,
    margin_seconds: float = 15.0,
) -> Path:
    manifest = read_manifest(manifest_path)
    segments = _load_segments(transcript_json)
    keyframes = manifest.get("keyframes", [])

    prev_t = 0.0
    for kf in keyframes:
        t = float(kf["time"])
        t0 = max(0.0, prev_t)
        t1 = t + margin_seconds
        text, ts0, ts1 = _text_in_window(segments, t0, t1)
        kf["transcript_text"] = text
        kf["transcript_start"] = ts0
        kf["transcript_end"] = ts1
        prev_t = t

    out_dir = Path(manifest_path).parent
    return write_manifest(out_dir, manifest.get("video", ""), keyframes,
                          montage=manifest.get("montage"))
