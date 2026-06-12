"""Lecturer-visit segmentation and board-change verification.

The snapshot triggers are event-based rather than signal-threshold-based:

  * VISIT END   - the lecturer was at a column for a sustained stretch and
                  walked away. If the column's content changed since the last
                  snapshot, capture it.
  * PRE-ERASE   - an erase is starting; capture the board's final state first.
  * FLUSH       - a long quiet stretch (or end of analysis) with pending,
                  uncaptured changes - a safety net for missed visits.

"Did it change" is answered by diffing the column's chalk-hold state against
the state at the previous snapshot. The diff is SHIFT-COMPENSATED: boards
slide vertically inside a column, so a pure slide (content moved, nothing
written) must not count as a change.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Visit:
    start: int  # analysis-frame index where sustained presence begins
    end: int    # analysis-frame index of the last presence frame


def max_pool(mask: np.ndarray, k: int, min_count: int = 1) -> np.ndarray:
    """Downsample a boolean mask by k; a cell is True if >= min_count pixels are.

    Pooling (rather than striding) preserves thin chalk strokes; min_count > 1
    suppresses single-pixel noise speckles.
    """
    if k <= 1:
        return mask.astype(bool)
    h, w = mask.shape
    h2, w2 = h // k, w // k
    if h2 == 0 or w2 == 0:
        return mask.astype(bool)
    counts = mask[: h2 * k, : w2 * k].reshape(h2, k, w2, k).sum(axis=(1, 3))
    return counts >= max(1, min_count)


def debounce_states(states: np.ndarray, k: int) -> np.ndarray:
    """Suppress cell flips that persist fewer than k consecutive frames.

    The lecturer's unmasked body edges toggle cells for a frame or two as he
    moves; chalk stays. Debouncing keeps the settle detector honest where he
    lingers.
    """
    if k <= 1 or len(states) == 0:
        return states
    out = np.empty_like(states)
    cur = states[0].copy()
    run = np.zeros(states.shape[1:], dtype=np.int16)
    out[0] = cur
    for i in range(1, len(states)):
        raw = states[i]
        disagree = raw != cur
        run[~disagree] = 0
        run[disagree] += 1
        flip = run >= k
        cur[flip] = raw[flip]
        run[flip] = 0
        out[i] = cur
    return out


def writing_window(change_curve: np.ndarray, floor: float) -> tuple[int, int]:
    """Estimate [start, end] indices of when content appeared, from the curve of
    cumulative change vs the reference state.

    Start = where the change first becomes non-trivial (10% of its final value,
    at least `floor`); end = where it reaches 90% of its final value.
    """
    n = len(change_curve)
    if n == 0:
        return 0, 0
    final = float(change_curve[-1])
    lo_thr = max(0.1 * final, floor)
    hi_thr = 0.9 * final
    start = next((i for i, v in enumerate(change_curve) if v >= lo_thr), 0)
    end = next((i for i in range(n) if change_curve[i] >= hi_thr), n - 1)
    return start, max(end, start)


def segment_visits(
    presence: np.ndarray,
    fps: float,
    min_seconds: float,
    bridge_seconds: float,
) -> list[Visit]:
    """Contiguous presence runs; gaps <= bridge merge, runs < min are dropped."""
    n = len(presence)
    bridge = max(1, round(bridge_seconds * fps))
    min_len = max(1, round(min_seconds * fps))

    runs: list[list[int]] = []
    i = 0
    while i < n:
        if not presence[i]:
            i += 1
            continue
        j = i
        while j < n and presence[j]:
            j += 1
        if runs and i - runs[-1][1] <= bridge:
            runs[-1][1] = j - 1
        else:
            runs.append([i, j - 1])
        i = j

    return [Visit(s, e) for s, e in runs if e - s + 1 >= min_len]


def _slide_compensated_ref(cur: np.ndarray, ref: np.ndarray, max_shift: int,
                           min_response: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference re-aligned for a dominant vertical slide, plus a validity mask."""
    valid = np.ones_like(cur)
    a = ref.astype(np.float32)
    b = cur.astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (_dx, dy), response = cv2.phaseCorrelate(a * win, b * win)
    shift = int(round(dy))
    if response < min_response or shift == 0 or abs(shift) > max_shift:
        return ref, valid
    shifted = np.roll(ref, shift, axis=0)
    if shift > 0:
        valid[:shift] = False
    else:
        valid[shift:] = False
    return shifted, valid


def compare_states(
    current: np.ndarray,
    reference: np.ndarray,
    max_shift: int = 0,
    min_response: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(added_mask, removed_mask, valid_mask) between two board states.

    With `max_shift` > 0 a dominant vertical slide is compensated first
    (content that merely moved counts as neither added nor removed), keeping
    whichever interpretation implies the smaller total change.
    """
    cur = current.astype(bool)
    ref = reference.astype(bool)
    if cur.shape != ref.shape or cur.size == 0:
        full = np.ones_like(cur)
        return full.copy(), full.copy(), full

    def measure(r: np.ndarray, valid: np.ndarray):
        add = cur & ~r & valid
        rem = ~cur & r & valid
        n = max(1, int(valid.sum()))
        return add, rem, (int(add.sum()) + int(rem.sum())) / n

    raw = measure(ref, np.ones_like(cur))
    best = (raw[0], raw[1], np.ones_like(cur))
    if max_shift > 0 and raw[2] > 0.0:
        sref, valid = _slide_compensated_ref(cur, ref, max_shift, min_response)
        comp = measure(sref, valid)
        if comp[2] < raw[2]:
            best = (comp[0], comp[1], valid)
    return best


def added_removed(
    current: np.ndarray,
    reference: np.ndarray,
    max_shift: int = 0,
    min_response: float = 0.10,
) -> tuple[float, float]:
    """(added, removed) chalk-cell fractions between two board states.

    Real writing is strongly ADD-dominant, eraser-smear drift is roughly
    symmetric, and a wipe is REMOVE-dominant - the split is what lets the
    caller tell them apart.
    """
    if current.shape != reference.shape or current.size == 0:
        return 1.0, 1.0
    add, rem, valid = compare_states(current, reference, max_shift, min_response)
    n = max(1, int(valid.sum()))
    return int(add.sum()) / n, int(rem.sum()) / n


def cohesion(cells: np.ndarray) -> float:
    """Fraction of True cells having >= 2 True 8-neighbours.

    Handwriting forms connected line-clusters (high cohesion ~0.7-0.95);
    drying eraser-smear streaks cross thresholds as scattered cells
    (~0.2-0.4). This is the cheapest reliable writing-vs-smear discriminator.
    """
    cells = cells.astype(bool)
    n = int(cells.sum())
    if n == 0:
        return 0.0
    neighbours = cv2.filter2D(cells.astype(np.uint8), -1,
                              np.ones((3, 3), np.uint8),
                              borderType=cv2.BORDER_CONSTANT) - cells
    return float((neighbours[cells] >= 2).mean())


def change_fraction(
    current: np.ndarray,
    reference: np.ndarray,
    max_shift: int = 0,
    min_response: float = 0.10,
) -> float:
    """Total changed-cell fraction (added + removed), slide-compensated."""
    add, rem = added_removed(current, reference, max_shift, min_response)
    return add + rem
