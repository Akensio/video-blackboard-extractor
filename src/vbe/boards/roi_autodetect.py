"""Assisted board-panel ROI detection.

On a median background plate the chalkboards are large, dark, roughly rectangular
regions (darker than the wooden frame and far darker than the lights). We threshold
for dark blobs, keep large rectangular contours, and emit ROIs in full-resolution
coordinates. This is a *seed* the user is expected to review/adjust, not a final
segmentation.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import ROI
from ..io.frame_cache import FrameCache


def reference_plate(cache: FrameCache, max_samples: int = 121) -> np.ndarray:
    """Global median plate (grayscale) over evenly-sampled cached frames."""
    n = cache.n
    idx = np.unique(np.linspace(0, n - 1, min(max_samples, n)).astype(int))
    stack = np.stack([cache.get(int(i)) for i in idx], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def _col_label(cx: float, width: int) -> str:
    third = width / 3.0
    return "left" if cx < third else ("center" if cx < 2 * third else "right")


def _row_label(cy: float, height: int) -> str:
    return "upper" if cy < height / 2.0 else "lower"


def autodetect_rois(
    plate_gray: np.ndarray,
    full_width: int,
    full_height: int,
    dark_threshold: int = 95,
    min_area_frac: float = 0.02,
    max_area_frac: float = 0.30,
) -> list[ROI]:
    """Detect dark rectangular board panels; return ROIs in full-resolution coords."""
    h, w = plate_gray.shape[:2]
    scale_x = full_width / w
    scale_y = full_height / h

    dark = (plate_gray < dark_threshold).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = h * w
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_frac * frame_area or area > max_area_frac * frame_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 0.08 * w or bh < 0.06 * h:
            continue
        boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: (b[1], b[0]))  # top-to-bottom, left-to-right
    rois: list[ROI] = []
    used: dict[str, int] = {}
    for (x, y, bw, bh) in boxes:
        cx, cy = x + bw / 2, y + bh / 2
        label = f"{_col_label(cx, w)}_{_row_label(cy, h)}"
        used[label] = used.get(label, 0) + 1
        name = label if used[label] == 1 else f"{label}_{used[label]}"
        fx0, fy0 = int(x * scale_x), int(y * scale_y)
        fx1, fy1 = int((x + bw) * scale_x), int((y + bh) * scale_y)
        rois.append(ROI(name=name, polygon=[(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)]))
    return rois
