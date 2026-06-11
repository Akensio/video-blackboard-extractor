"""timeline.json read/write: the per-video board timeline (and paired text)."""
from __future__ import annotations

import json
from pathlib import Path

TIMELINE_NAME = "timeline.json"


def write_timeline(out_dir: str | Path, payload: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / TIMELINE_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_timeline(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
