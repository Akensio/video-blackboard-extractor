"""Temporal-median background reconstruction.

A pixel that is background more than ~half the time survives a temporal median,
while the moving lecturer washes out. We use:
  * a coarse grid of analysis-resolution plates as the occlusion reference, and
  * (optionally) a person-mask-gated median for the case where the lecturer
    stands still while writing, which breaks the plain-median assumption.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.frame_cache import FrameCache  # noqa: F401  (type hint convenience)


def median_plate(frames: np.ndarray) -> np.ndarray:
    """Per-pixel median over a stack of frames -> uint8 plate."""
    if frames.shape[0] == 0:
        raise ValueError("median_plate received zero frames")
    return np.median(frames, axis=0).astype(np.uint8)


def masked_median_plate(frames: np.ndarray, person_masks: np.ndarray) -> np.ndarray:
    """Median over only the frames where each pixel is NOT person-masked.

    `person_masks` is a boolean array shaped like `frames` (True = lecturer).
    Pixels masked in every frame fall back to the plain median.
    """
    masked = np.ma.array(frames, mask=person_masks)
    med = np.ma.median(masked, axis=0)
    plate = med.filled(np.nan)
    # Fallback for fully-masked pixels.
    holes = np.isnan(plate)
    if holes.any():
        plain = np.median(frames, axis=0)
        plate[holes] = plain[holes]
    return plate.astype(np.uint8)


def _sample_indices(lo: int, hi: int, max_samples: int) -> np.ndarray:
    count = hi - lo
    if count <= max_samples:
        return np.arange(lo, hi)
    return np.unique(np.linspace(lo, hi - 1, max_samples).astype(int))


@dataclass
class PlateGrid:
    """Coarse grid of analysis-resolution background plates."""

    centers: np.ndarray       # (G,) frame indices
    plates: np.ndarray        # (G, H, W) uint8

    def nearest(self, i: int) -> np.ndarray:
        g = int(np.argmin(np.abs(self.centers - i)))
        return self.plates[g]


def build_plate_grid(
    cache: FrameCache,
    half_window_frames: int,
    stride_frames: int | None = None,
    max_samples: int = 31,
) -> PlateGrid:
    """Compute centered median plates every `stride_frames` for occlusion reference."""
    half = max(1, half_window_frames)
    stride = stride_frames or max(1, half // 2)
    centers = list(range(0, cache.n, stride))
    if centers[-1] != cache.n - 1:
        centers.append(cache.n - 1)

    plates = np.empty((len(centers), cache.height, cache.width), dtype=np.uint8)
    for g, c in enumerate(centers):
        lo, hi = cache.window_indices(c, half)
        idx = _sample_indices(lo, hi, max_samples)
        stack = np.stack([cache.get(int(j)) for j in idx], axis=0)
        plates[g] = median_plate(stack)
    return PlateGrid(centers=np.array(centers), plates=plates)
