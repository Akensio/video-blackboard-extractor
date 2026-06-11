"""Writing-burst segmentation of a board's chalk-fullness signal.

A lecture board alternates between three regimes:
  * writing  - fullness rises (the lecturer is adding chalk),
  * talking  - fullness is flat (he stepped back to explain),
  * erase    - fullness drops sharply (wipe or board slid out of its column).

A *burst* is a maximal writing stretch; short talking pauses inside it are
merged (config `burst_merge_gap_seconds`), but bursts never merge across an
erase. Each burst yields one timeline snapshot whose `writing_interval` is the
burst's [start, end] - exactly the "half board during [a,b], full board during
[c,d]" timeline the notes need.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class Burst:
    start: int   # analysis-frame index where writing begins
    end: int     # analysis-frame index where writing stops
    gain: float  # fullness added over the burst


def fill_nan(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN samples (occluded readings) from neighbours."""
    v = np.asarray(values, dtype=np.float64).copy()
    bad = np.isnan(v)
    if not bad.any():
        return v
    if bad.all():
        return np.zeros_like(v)
    idx = np.arange(len(v))
    v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return v


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def writing_rate(fullness: np.ndarray, fps: float, deriv_window_seconds: float = 31.0) -> np.ndarray:
    """Smoothed derivative of fullness in fullness-units per SECOND."""
    n = len(fullness)
    if n < 7:
        return np.zeros(n)
    win = _odd(min(max(7, round(deriv_window_seconds * fps)), n - 1 if n % 2 == 0 else n))
    if win > n:
        win = _odd(n - 1)
    df = savgol_filter(fullness, win, polyorder=2, deriv=1)
    return df * fps


def _contiguous_regions(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end] index pairs (inclusive) of runs of True."""
    out: list[tuple[int, int]] = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def detect_bursts(
    fullness: np.ndarray,
    fps: float,
    rate_threshold: float,
    merge_gap_seconds: float,
    min_gain: float,
    erase_indices: list[int] | None = None,
    deriv_window_seconds: float = 31.0,
) -> list[Burst]:
    """Segment a (smoothed, NaN-free) fullness signal into writing bursts.

    `rate_threshold` is in fullness-units/second. Bursts separated by a pause
    shorter than `merge_gap_seconds` are merged, unless an erase event sits in
    the gap. Bursts that add less than `min_gain` fullness are dropped.
    """
    n = len(fullness)
    if n == 0:
        return []
    erase_set = sorted(erase_indices or [])

    rate = writing_rate(fullness, fps, deriv_window_seconds)
    regions = _contiguous_regions(rate > rate_threshold)
    if not regions:
        return []

    def erase_between(a: int, b: int) -> bool:
        return any(a < e <= b for e in erase_set)

    merge_gap = max(1, round(merge_gap_seconds * fps))
    merged: list[list[int]] = [list(regions[0])]
    for s, e in regions[1:]:
        prev_end = merged[-1][1]
        if s - prev_end <= merge_gap and not erase_between(prev_end, s):
            merged[-1][1] = e
        else:
            merged.append([s, e])

    bursts: list[Burst] = []
    for s, e in merged:
        gain = float(fullness[e] - fullness[s])
        if gain >= min_gain:
            bursts.append(Burst(start=s, end=e, gain=gain))
    return bursts
