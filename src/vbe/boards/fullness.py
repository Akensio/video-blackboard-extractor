"""Chalk detection and the per-ROI "fullness" signal.

Chalk is thin and bright on a dark board. A morphological white top-hat keeps
small bright structures (strokes) and suppresses large uniform regions - both
the board itself and the lecturer's solid body - which makes the signal fairly
robust to the moving lecturer even before background reconstruction.
"""
from __future__ import annotations

import cv2
import numpy as np


def chalk_mask(gray: np.ndarray, tophat_kernel: int, threshold: int) -> np.ndarray:
    """Boolean mask of chalk-like bright thin strokes in a grayscale frame."""
    k = max(3, tophat_kernel | 1)  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return tophat > threshold


def fullness_in_mask(gray: np.ndarray, region_mask: np.ndarray,
                     tophat_kernel: int, threshold: int) -> float:
    """Fraction of the region covered by chalk."""
    area = int(region_mask.sum())
    if area == 0:
        return 0.0
    chalk = chalk_mask(gray, tophat_kernel, threshold)
    return float(np.count_nonzero(chalk & region_mask)) / area


def laplacian_sharpness(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Variance of the Laplacian over a bbox - higher means sharper (less blur)."""
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = gray[y0:y1, x0:x1]
    return float(cv2.Laplacian(region, cv2.CV_64F).var())


def smooth_signal(values: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay smoothing; falls back gracefully for short signals."""
    from scipy.signal import savgol_filter

    n = len(values)
    if n < 5:
        return values.astype(np.float64)
    win = min(window | 1, n if n % 2 == 1 else n - 1)
    win = max(5, win)
    if win > n:
        return values.astype(np.float64)
    polyorder = min(2, win - 1)
    return savgol_filter(values.astype(np.float64), win, polyorder)
