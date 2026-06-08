"""Board region-of-interest helpers.

ROIs are stored in FULL-resolution coordinates (see config.ROI). Analysis runs
at a downscaled resolution, so we scale polygons by `analysis_width/full_width`.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import ROI


def analysis_scale(full_width: int, analysis_width: int) -> float:
    if analysis_width >= full_width:
        return 1.0
    return analysis_width / full_width


def default_full_roi(full_width: int, full_height: int, inset: float = 0.0) -> ROI:
    """Whole-frame ROI (optionally inset by a fraction on each side)."""
    dx = int(full_width * inset)
    dy = int(full_height * inset)
    return ROI(
        name="full",
        polygon=[
            (dx, dy),
            (full_width - dx, dy),
            (full_width - dx, full_height - dy),
            (dx, full_height - dy),
        ],
    )


def scaled_polygon(roi: ROI, scale: float) -> np.ndarray:
    pts = np.array(roi.polygon, dtype=np.float64) * scale
    return np.round(pts).astype(np.int32)


def roi_mask(roi: ROI, height: int, width: int, scale: float) -> np.ndarray:
    """Boolean mask (height, width) for the ROI at analysis resolution."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [scaled_polygon(roi, scale)], 255)
    return mask.astype(bool)


def scaled_bbox(roi: ROI, scale: float, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi.bbox()
    x0 = max(0, int(np.floor(x0 * scale)))
    y0 = max(0, int(np.floor(y0 * scale)))
    x1 = min(width, int(np.ceil(x1 * scale)))
    y1 = min(height, int(np.ceil(y1 * scale)))
    return x0, y0, x1, y1


def fullres_bbox(roi: ROI, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi.bbox()
    return (max(0, x0), max(0, y0), min(width, x1), min(height, y1))


def panel_rois(rois: list[ROI]) -> list[ROI]:
    """ROIs excluding the synthetic full-wall ROI."""
    return [r for r in rois if r.name != "full"]
