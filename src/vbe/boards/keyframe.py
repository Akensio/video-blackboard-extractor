"""Keyframe scoring and full-resolution clean-frame export.

For a chosen "fullest" moment we export a full-resolution frame with the
lecturer composited out: take the real frame at that instant (exact, latest
content) and patch only the lecturer's pixels with a backward temporal-median
plate (which contains the board behind him from nearby times).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..io import ffmpeg
from . import background, person


def normalize(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def score_frames(
    fullness: np.ndarray,
    occlusion: np.ndarray,
    sharpness: np.ndarray,
    w_fullness: float,
    w_occlusion: float,
    w_sharpness: float,
) -> np.ndarray:
    """Per-frame keyframe score (higher is better)."""
    f = normalize(fullness)
    s = normalize(sharpness)
    return w_fullness * f - w_occlusion * np.asarray(occlusion, float) + w_sharpness * s


def select_best_index(scores: np.ndarray, lo: int, hi: int) -> int:
    """Argmax of `scores` within [lo, hi)."""
    lo = max(0, lo)
    hi = min(len(scores), hi)
    if hi <= lo:
        return lo
    return lo + int(np.argmax(scores[lo:hi]))


def build_clean_frame(
    video_path,
    t: float,
    window_seconds: float,
    fps: float,
    global_plate: np.ndarray | None = None,
    fg_thr: int = 38,
    open_ksize: int = 11,
    dilate_px: int = 11,
    segmenter: "person.YoloPersonSegmenter | None" = None,
) -> np.ndarray:
    """Return a full-resolution BGR keyframe at time `t` with the lecturer removed.

    Strategy:
      * the real frame at `t` provides exact, latest chalk content;
      * a backward temporal-median plate [t-window, t] supplies the board behind
        the lecturer (he has usually moved within the window);
      * where he *lingered* the whole window the local plate still contains him,
        so those residual pixels are filled from a person-free global plate (the
        whole-lecture median). This keeps the default path ML-free yet clean.
    """
    start = max(0.0, t - window_seconds)
    duration = t - start
    frames = [f for _, f in ffmpeg.decode_frames(
        video_path, fps=fps, start=start, duration=max(duration, 1.0 / fps),
        size=None, gray=False)]
    if not frames:
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.gettempdir()) / "vbe_single.png"
        ffmpeg.extract_frame(video_path, t, tmp, gray=False)
        return cv2.imread(str(tmp))

    raw = frames[-1]  # closest to t
    # Subsample the window to bound memory/cost of the per-pixel median.
    sample = _subsample(frames, max_n=41)
    stack = np.stack(sample, axis=0)

    if segmenter is not None:
        # Person-masked median: exclude the lecturer per-pixel, so the infill
        # source itself is person-free even where he lingered.
        masks = np.stack([segmenter.mask(f, dilate_px=dilate_px) for f in sample], axis=0)
        plate = background.masked_median_plate(stack, np.repeat(masks[..., None], 3, axis=3))
        pmask = segmenter.mask(raw, dilate_px=dilate_px)
    else:
        plate = background.median_plate(stack)
        # Detect the lecturer as the large blob differing from a clean reference;
        # the morphological open in person_foreground discards thin chalk diffs.
        ref = cv2.cvtColor(global_plate, cv2.COLOR_BGR2GRAY) if global_plate is not None \
            else cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        pmask = person.person_foreground(
            cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), ref,
            thr=fg_thr, open_ksize=open_ksize, dilate_px=dilate_px,
        )

    out = raw.copy()
    out[pmask] = plate[pmask]

    # Fill any pixels where the local plate still contains the lecturer (he
    # lingered the whole window) from the person-free global plate.
    if global_plate is not None:
        gloc = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gglob = cv2.cvtColor(global_plate, cv2.COLOR_BGR2GRAY)
        residual = pmask & (cv2.absdiff(gloc, gglob) > fg_thr)
        out[residual] = global_plate[residual]

    return out


def _subsample(items: list, max_n: int) -> list:
    if len(items) <= max_n:
        return items
    idx = np.linspace(0, len(items) - 1, max_n).astype(int)
    return [items[i] for i in idx]
