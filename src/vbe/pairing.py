"""Pair timeline snapshots with the transcript.

The lecturer talks about a board while writing it and just after, and he often
interleaves writing on several boards, so windows are allowed to OVERLAP: each
snapshot gets the speech from the start of its own writing burst until either
its capture (+margin) or the start of the next snapshot's burst - whichever is
later. Concurrently-written boards therefore share the text spoken while both
were in progress; that redundancy is deliberate and harmless for notes.
"""
from __future__ import annotations

import json
from pathlib import Path

from .manifest import read_timeline, write_timeline


def _load_segments(transcript_json: str | Path) -> list[dict]:
    data = json.loads(Path(transcript_json).read_text(encoding="utf-8"))
    return data.get("segments", [])


def _text_in_window(segments: list[dict], t0: float, t1: float) -> tuple[str, float | None, float | None]:
    picked = [s for s in segments if s["end"] >= t0 and s["start"] <= t1]
    if not picked:
        return "", None, None
    text = " ".join(s["text"].strip() for s in picked).strip()
    return text, picked[0]["start"], picked[-1]["end"]


def pair_timeline(
    timeline_path: str | Path,
    transcript_json: str | Path,
    margin_seconds: float = 15.0,
    last_window_seconds: float = 300.0,
) -> Path:
    timeline = read_timeline(timeline_path)
    segments = _load_segments(transcript_json)
    snaps = sorted(timeline.get("snapshots", []),
                   key=lambda s: s["writing_interval"][0])

    for i, snap in enumerate(snaps):
        t0 = max(0.0, snap["writing_interval"][0] - margin_seconds)
        own_span = snap["capture_time"] + margin_seconds
        if i + 1 < len(snaps):
            t1 = max(own_span, snaps[i + 1]["writing_interval"][0])
        else:
            # Last snapshot: bounded window (not the entire remaining lecture),
            # capped at the erase if one follows.
            t1 = snap["capture_time"] + last_window_seconds
            if snap.get("erased_at"):
                t1 = min(t1, snap["erased_at"] + margin_seconds)
        text, ts0, ts1 = _text_in_window(segments, t0, t1)
        snap["transcript"] = {"start": ts0, "end": ts1, "text": text}

    # Speech before the first writing burst (course intro, recap, etc.).
    if snaps:
        first_start = snaps[0]["writing_interval"][0]
        pre_text, p0, p1 = _text_in_window(segments, 0.0, first_start)
        if pre_text:
            timeline["preamble_transcript"] = {"start": p0, "end": p1, "text": pre_text}

    timeline["snapshots"] = sorted(snaps, key=lambda s: s["capture_time"])
    timeline["transcript_file"] = Path(transcript_json).name
    out_dir = Path(timeline_path).parent
    return write_timeline(out_dir, timeline)
