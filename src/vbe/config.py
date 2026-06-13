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
    analysis_person_mask: str = "auto"  # auto | yolo | heuristic (analysis-pass mask)
    analysis_seg_stride_seconds: float = 4.0  # YOLO sampling stride for the analysis mask
    median_window_seconds: float = 90.0
    seg_model: str = "yolov8n-seg.pt"
    person_dilate_px: int = 9

    # --- chalk fullness signal ---
    chalk_tophat_kernel: int = 15
    chalk_threshold: int = 18
    fullness_smooth_window: int = 15

    # --- snapshot triggering (change + settle) ---
    snapshot_change_min: float = 0.006   # changed fraction of the board that earns a snapshot
    # (the emit gate counts COHESIVE added cells against snapshot_change_min;
    # scattered smear/slide-residue cells contribute ~nothing by construction)
    stable_seconds: float = 45.0         # writing pause that marks content as "settled"
    stable_eps: float = 0.010            # state change below this over stable_seconds = settled
    chalk_on_threshold: int = 30         # tophat level that turns a state pixel ON
    chalk_off_threshold: int = 14        # tophat level that turns a state pixel OFF
    state_downsample: int = 4            # chalk-state grid downsample for change diffs
    state_min_count: int = 2             # pixels per pooled cell required (noise suppression)
    state_debounce_frames: int = 3       # cell flips must persist this many frames
    capture_max_occlusion: float = 0.25  # defer capture while board is this occluded
    # Vertical slide compensation for change diffs, in pooled cells. 0 = off:
    # with one ROI per physical board (boards erased in place, never slid) a
    # slide cannot move content within a board, so compensation can only mask
    # real writing. Raise only for halls where boards genuinely slide.
    change_slide_max_cells: int = 0

    # --- lecturer presence (informational `visits` field only) ---
    presence_threshold: float = 0.06     # board occlusion above this = he is at the board
    presence_min_seconds: float = 4.0
    presence_bridge_seconds: float = 8.0

    # --- snapshot capture windows ---
    capture_post_seconds: float = 30.0       # search window after a trigger for the cleanest frame
    capture_preerase_back_seconds: float = 20.0  # backward window before an erase peak

    # --- full-res export ---
    export_window_seconds: float = 60.0  # backward median window for clean-frame export
    export_fps: float = 0.5              # decode fps inside that window
    enhance_crops: bool = True           # also emit CLAHE-enhanced grayscale crops
    trim_crops: bool = False             # content-aware crop trim (off: never risk cutting chalk)

    # --- erase detection ---
    erase_min_drop: float = 0.25           # RELATIVE drop from the running peak
    erase_refractory_seconds: float = 20.0
    erase_min_peak: float = 0.03           # boards emptier than this never "erase"
    erase_persist_seconds: float = 10.0    # the drop must hold this long
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
