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
    min_peak: float = 0.03,
    persist_frames: int = 8,
) -> list[Event]:
    """Find erase events: RELATIVE drops from a running peak that persist.

    `min_drop` is a fraction of the running peak (a wipe removes most of a
    board, so ~0.35+ of its chalk disappears), not an absolute fullness value -
    full boards only reach ~0.15-0.2 absolute fullness, so absolute thresholds
    near that scale can never fire. To avoid firing on transient dips (residual
    occlusion noise), the signal must stay below the drop level for
    `persist_frames` consecutive samples, and near-empty boards
    (peak < `min_peak`) never produce events.
    """
    events: list[Event] = []
    n = len(fullness)
    if n == 0:
        return events

    peak_val = float(fullness[0])
    peak_idx = 0
    last_event = -(10**9)

    i = 1
    while i < n:
        v = float(fullness[i])
        if v >= peak_val:
            peak_val = v
            peak_idx = i
            i += 1
            continue
        dropped = (peak_val >= min_peak
                   and (peak_val - v) >= min_drop * peak_val
                   and (i - last_event) >= refractory_frames)
        if dropped:
            # require persistence: the drop must hold for persist_frames samples
            hold = fullness[i:i + persist_frames]
            threshold = peak_val - min_drop * peak_val
            if len(hold) >= max(1, persist_frames // 2) and np.all(hold <= threshold):
                events.append(Event(drop_index=i, peak_index=peak_idx,
                                    drop=peak_val - v))
                last_event = i
                # reset the running peak to the post-drop level
                j = min(i + persist_frames, n - 1)
                peak_val = float(fullness[j])
                peak_idx = j
                i = j + 1
                continue
        i += 1
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
