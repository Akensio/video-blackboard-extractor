"""manifest.json read/write: the per-video index of keyframes (and their paired text)."""
from __future__ import annotations

import json
from pathlib import Path


def write_manifest(out_dir: str | Path, video: str, keyframes: list[dict],
                   montage: str | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    payload = {"video": video, "montage": montage, "keyframes": keyframes}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
