"""Legibility enhancement for board crops.

Chalk on a dark board photographed at 1080p is low-contrast; CLAHE plus a mild
unsharp mask makes strokes far easier for humans and vision models to read.
The enhanced image is emitted *alongside* the raw crop, never instead of it.
"""
from __future__ import annotations

import cv2
import numpy as np


def trim_to_board(
    bgr: np.ndarray,
    board_p25_threshold: int = 118,
    pad: int = 14,
) -> np.ndarray:
    """Trim bright wood/screen margins so the crop hugs the dark board surface.

    A line (row or column) is board-like when its 25th-percentile brightness is
    below `board_p25_threshold`: even with light-bar glare or dense chalk, most
    board pixels stay dark, while wood paneling and the projector screen sit
    well above it. The crop is reduced to the longest board-like run per axis;
    short bright interruptions (board seams, chalk rails) are bridged so a
    two-board stack counts as one band. A safety check refuses pathological
    trims (anything below a quarter of the original size).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Chalk strokes mark a line as board even when chalk-dust smears or glare
    # push its brightness up (freshly erased boards are noticeably lighter).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    chalk = tophat > 18

    def board_span(p25: np.ndarray, chalk_frac: np.ndarray) -> tuple[int, int]:
        """Span from the first to the last SIGNIFICANT board-like run.

        Sliding boards can sit apart with a wide wood gap between them; keeping
        the full span (gap included) guarantees no board's content is ever
        trimmed away. Tiny runs (shadow slivers, dark trim strips) don't anchor
        the span.
        """
        n = len(p25)
        boardlike = (p25 < board_p25_threshold) | (chalk_frac > 0.004)
        min_run = max(40, n // 10)
        runs: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if not boardlike[i]:
                i += 1
                continue
            j = i
            while j < n and boardlike[j]:
                j += 1
            if j - i >= min_run:
                runs.append((i, j))
            i = j
        if not runs:
            return 0, n
        return runs[0][0], runs[-1][1]

    y0, y1 = board_span(np.percentile(gray, 25, axis=1), chalk.mean(axis=1))
    x0, x1 = board_span(np.percentile(gray, 25, axis=0), chalk.mean(axis=0))
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad)
    if y1 - y0 < h // 4 or x1 - x0 < w // 4:  # safety: refuse pathological trims
        return bgr
    return bgr[y0:y1, x0:x1]


def find_rail(col_gray: np.ndarray, margin: int = 35,
              interior: tuple[float, float] = (0.18, 0.82)) -> int | None:
    """Row (local to the crop) of the wooden rail splitting a column's two boards.

    The rail is a horizontal wooden strip: bright across the FULL width, so its
    per-row MEDIAN brightness stands well above the dark board baseline (chalk
    is patchy and barely moves the median). Returns the strongest such interior
    ridge, or None if no rail clears `margin` (one board fills the column, or it
    is blank). Detected per-frame, so it tracks the boards as they slide.
    """
    med = np.median(col_gray, axis=1)
    base = float(np.percentile(med, 30))
    n = len(med)
    lo, hi = int(interior[0] * n), int(interior[1] * n)
    if hi <= lo:
        return None
    rel = med[lo:hi] - base
    j = int(np.argmax(rel))
    if rel[j] < margin:
        return None
    return lo + j


def board_cell_extent(
    col_gray: np.ndarray,
    y_lo: float,
    y_hi: float,
    board_p25_threshold: int = 118,
    bridge: int = 18,
    pad: int = 8,
) -> tuple[int, int]:
    """Vertical extent of the single board panel that contains new writing.

    Given a clean column crop and the row band [y_lo, y_hi] where fresh chalk
    appeared, return the [top, bottom] of the dark board surface enclosing that
    band - i.e. snap the writing to the wooden rails above and below it. This
    isolates ONE board even when boards have slid, because it follows the dark
    surface the chalk actually sits on rather than any fixed line. A row is
    board-like when its 25th-percentile brightness is below the threshold
    (wood/rails sit well above it); short bright interruptions are bridged.
    """
    n = col_gray.shape[0]
    y_lo = max(0, min(n - 1, int(y_lo)))
    y_hi = max(1, min(n, int(y_hi)))
    p25 = np.percentile(col_gray, 25, axis=1)
    boardlike = p25 < board_p25_threshold

    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not boardlike[i]:
            i += 1
            continue
        j, gap, last = i, 0, i
        while j < n:
            if boardlike[j]:
                last, gap = j, 0
            else:
                gap += 1
                if gap > bridge:
                    break
            j += 1
        runs.append((i, last + 1))
        i = j

    def overlap(r: tuple[int, int]) -> int:
        return max(0, min(r[1], y_hi) - max(r[0], y_lo))

    if not runs:
        return max(0, y_lo - pad), min(n, y_hi + pad)
    best = max(runs, key=overlap)
    if overlap(best) <= 0:  # band fell in wood (shouldn't happen) - pad the band
        return max(0, y_lo - pad), min(n, y_hi + pad)
    return max(0, best[0] - pad), min(n, best[1] + pad)


def enhance_board_crop(
    bgr: np.ndarray,
    clip_limit: float = 3.0,
    tile: int = 8,
    unsharp_sigma: float = 1.2,
    unsharp_amount: float = 0.5,
) -> np.ndarray:
    """Grayscale CLAHE + unsharp masking; returns a single-channel uint8 image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (0, 0), unsharp_sigma)
    sharp = cv2.addWeighted(eq, 1.0 + unsharp_amount, blur, -unsharp_amount, 0)
    return sharp
