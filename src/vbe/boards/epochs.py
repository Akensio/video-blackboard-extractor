"""Segment a board's life into epochs by detecting erase / slide events.

The chalk-fullness signal grows during a writing burst, then drops sharply when
the board is erased or slid away. Each such drop ends an epoch; the frame near
the preceding fullness *peak* is the moment the board was most complete - the
keyframe we want to capture.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Event:
    drop_index: int    # where the drop is detected
    peak_index: int    # the preceding fullness peak (the fullest moment)
    drop: float        # magnitude of the fullness drop
    kind: str = "erase"  # 'erase' or 'slide' (classified later with image data)


def detect_events(
    fullness: np.ndarray,
    min_drop: float,
    refractory_frames: int,
) -> list[Event]:
    """Find significant drops from a running peak (a state machine over the signal)."""
    events: list[Event] = []
    n = len(fullness)
    if n == 0:
        return events

    peak_val = float(fullness[0])
    peak_idx = 0
    last_event = -(10**9)

    for i in range(1, n):
        v = float(fullness[i])
        if v >= peak_val:
            peak_val = v
            peak_idx = i
            continue
        if peak_val - v >= min_drop and (i - last_event) >= refractory_frames:
            events.append(Event(drop_index=i, peak_index=peak_idx, drop=peak_val - v))
            last_event = i
            peak_val = v
            peak_idx = i
    return events


def capture_indices(fullness: np.ndarray, events: list[Event], min_fullness: float = 0.0) -> list[int]:
    """Indices of the fullest moments to capture: each pre-erase peak + the final state."""
    n = len(fullness)
    idxs = [e.peak_index for e in events if fullness[e.peak_index] > min_fullness]

    # Trailing epoch (after the last event): capture its max-fullness frame.
    start = (events[-1].drop_index if events else 0)
    if start < n:
        tail = fullness[start:]
        if len(tail) and float(tail.max()) > min_fullness:
            idxs.append(start + int(np.argmax(tail)))

    return sorted(set(idxs))


def classify_slide(prev_roi: np.ndarray, next_roi: np.ndarray, shift_threshold_px: float) -> tuple[bool, float]:
    """Detect a vertical slide between two grayscale ROI patches via phase correlation.

    Returns (is_slide, vertical_shift_px). A slide moves content vertically;
    an erase does not.
    """
    if prev_roi.shape != next_roi.shape or prev_roi.size == 0:
        return False, 0.0
    a = prev_roi.astype(np.float32)
    b = next_roi.astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (_, dy), _resp = cv2.phaseCorrelate(a * win, b * win)
    return abs(dy) >= shift_threshold_px, float(dy)
