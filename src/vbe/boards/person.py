"""Lecturer occlusion estimation.

Default (no-ML) path: the lecturer is the large blob that differs from the local
median plate. A morphological opening removes thin chalk differences, leaving the
solid body, which we measure per ROI. An optional YOLO segmenter ([seg] extra)
gives crisper person masks for the gated-median export path.
"""
from __future__ import annotations

import cv2
import numpy as np


def person_foreground(
    gray: np.ndarray,
    plate: np.ndarray,
    thr: int = 38,
    open_ksize: int = 7,
    dilate_px: int = 0,
) -> np.ndarray:
    """Boolean mask of the lecturer (large foreground blob) vs the plate."""
    diff = cv2.absdiff(gray, plate)
    fg = (diff > thr).astype(np.uint8)
    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        fg = cv2.dilate(fg, k)
    return fg.astype(bool)


def occlusion_in_mask(person_mask: np.ndarray, region_mask: np.ndarray) -> float:
    area = int(region_mask.sum())
    if area == 0:
        return 0.0
    return float(np.count_nonzero(person_mask & region_mask)) / area


class YoloPersonSegmenter:
    """Optional person segmentation via ultralytics YOLO ([seg] extra).

    Operates on BGR frames (any resolution): used by the gated-median export
    and, sampled, by the analysis-time mask.
    """

    def __init__(self, model_name: str = "yolov8n-seg.pt"):
        from ultralytics import YOLO  # imported lazily; only when [seg] installed

        self.model = YOLO(model_name)

    def mask(self, frame_bgr: np.ndarray, dilate_px: int = 9) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        res = self.model.predict(frame_bgr, classes=[0], verbose=False)[0]
        out = np.zeros((h, w), dtype=np.uint8)
        if res.masks is not None:
            for m in res.masks.data.cpu().numpy():
                mm = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                out |= (mm > 0.5).astype(np.uint8)
        if dilate_px > 0 and out.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
            out = cv2.dilate(out, k)
        return out.astype(bool)


class SampledPersonMask:
    """YOLO person masks on a coarse time grid for the analysis pass.

    Running the segmenter on every analysis frame is wasteful - the lecturer
    moves slowly relative to 1 fps. Masks are computed every `stride` frames
    and a query returns the UNION of the two samples bracketing the frame, so
    his position anywhere between the samples is covered. Lazy + cached, so
    cost is ~n/stride inferences per video.
    """

    def __init__(self, cache, segmenter: YoloPersonSegmenter,
                 stride: int, dilate_px: int = 9):
        self.cache = cache
        self.segmenter = segmenter
        self.stride = max(1, stride)
        self.dilate_px = dilate_px
        self._samples: dict[int, np.ndarray] = {}

    def _sample(self, slot: int) -> np.ndarray:
        if slot not in self._samples:
            i = min(slot * self.stride, self.cache.n - 1)
            bgr = cv2.cvtColor(self.cache.get(i), cv2.COLOR_GRAY2BGR)
            self._samples[slot] = self.segmenter.mask(bgr, dilate_px=self.dilate_px)
        return self._samples[slot]

    def mask(self, i: int) -> np.ndarray:
        slot = i // self.stride
        m = self._sample(slot)
        if (slot + 1) * self.stride < self.cache.n:
            m = m | self._sample(slot + 1)
        return m
