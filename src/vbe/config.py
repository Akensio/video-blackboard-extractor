"""Configuration models and YAML loading/merging.

Precedence (lowest to highest):
  1. pydantic model defaults (authoritative fallback, mirror of configs/default.yaml)
  2. configs/default.yaml (if found next to the repo)
  3. the per-video config file passed on the CLI
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ROI(BaseModel):
    """A board panel region, in FULL-RESOLUTION pixel coordinates.

    `polygon` is a list of [x, y] points (>=3). Rectangles are just 4-point
    polygons. The bounding box is used for cropping; the filled polygon is used
    for masking during analysis.
    """

    name: str
    polygon: list[tuple[int, int]]

    def bbox(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)


class Config(BaseModel):
    # --- frame decoding / analysis ---
    analysis_fps: float = 1.0
    analysis_width: int = 960
    export_format: str = "png"

    # --- lecturer removal ---
    person_removal: str = "median"  # median | gated_median | none
    median_window_seconds: float = 90.0
    seg_model: str = "yolov8n-seg.pt"
    person_dilate_px: int = 9

    # --- chalk fullness signal ---
    chalk_tophat_kernel: int = 15
    chalk_threshold: int = 18
    fullness_smooth_window: int = 9

    # --- epoch segmentation ---
    erase_min_drop: float = 0.18
    erase_refractory_seconds: float = 20.0
    slide_shift_px: float = 25.0
    min_epoch_seconds: float = 25.0

    # --- keyframe scoring ---
    keyframe_presnap_seconds: float = 8.0
    weight_fullness: float = 1.0
    weight_occlusion: float = 1.5
    weight_sharpness: float = 0.4
    emit_region_crops: bool = True

    # --- dedup ---
    phash_max_distance: int = 8
    ssim_tiebreak: float = 0.92

    # --- transcription ---
    asr_model: str = "large-v3"
    asr_device: str = "auto"  # auto | cuda | cpu
    asr_compute_type_gpu: str = "int8_float16"
    asr_compute_type_cpu: str = "int8"
    asr_beam_size_gpu: int = 2
    asr_beam_size_cpu: int = 5
    asr_language: str = "en"
    asr_vad: bool = True
    asr_word_timestamps: bool = True
    asr_condition_on_previous_text: bool = True
    asr_initial_prompt: str = (
        "This is a graduate quantum field theory lecture. Terms used include "
        "Lagrangian, Hamiltonian, path integral, propagator, Green's function, "
        "perturbation theory, Feynman diagram, vertex, loop integral, self-energy, "
        "counterterm, regularization, dimensional regularization, gauge field, "
        "Ward identity, beta function, renormalization group, renormalization."
    )

    # --- pairing ---
    pair_margin_seconds: float = 15.0

    # --- per-video board regions ---
    rois: list[ROI] = Field(default_factory=list)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at top level.")
    return data


def find_default_config() -> Path | None:
    """Locate configs/default.yaml by walking up from this module."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "configs" / "default.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_config(config_path: str | Path | None = None) -> Config:
    """Build a Config from the bundled defaults plus an optional per-video file."""
    merged: dict[str, Any] = {}

    default_path = find_default_config()
    if default_path is not None:
        merged.update(_read_yaml(default_path))

    if config_path is not None:
        merged.update(_read_yaml(Path(config_path)))

    return Config(**merged)


def dump_config(cfg: Config, path: str | Path) -> None:
    """Write a config (including ROIs) to YAML, preserving comments-free fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump()
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
