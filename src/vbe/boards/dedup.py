"""Near-duplicate keyframe suppression via perceptual hash + SSIM tiebreak."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import imagehash
import numpy as np
from PIL import Image


def phash_bgr(image_bgr: np.ndarray) -> imagehash.ImageHash:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def _ssim(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim

    h = min(a_bgr.shape[0], b_bgr.shape[0])
    w = min(a_bgr.shape[1], b_bgr.shape[1])
    a = cv2.cvtColor(cv2.resize(a_bgr, (w, h)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(b_bgr, (w, h)), cv2.COLOR_BGR2GRAY)
    return float(ssim(a, b))


@dataclass
class KeyframeCandidate:
    index: int            # analysis-frame index
    time: float
    image: np.ndarray     # BGR full-res clean keyframe
    fullness: float
    occlusion: float
    meta: dict


def deduplicate(
    candidates: list[KeyframeCandidate],
    phash_max_distance: int,
    ssim_tiebreak: float,
) -> list[KeyframeCandidate]:
    """Keep one representative per near-duplicate group, preferring fuller boards.

    Candidates are processed in time order; a new candidate that is a near
    duplicate of a kept one replaces it iff it is fuller, otherwise it is dropped.
    """
    kept: list[KeyframeCandidate] = []
    hashes: list[imagehash.ImageHash] = []

    for cand in sorted(candidates, key=lambda c: c.time):
        h = phash_bgr(cand.image)
        dup_pos = -1
        for j, hk in enumerate(hashes):
            if (h - hk) <= phash_max_distance and _ssim(cand.image, kept[j].image) >= ssim_tiebreak:
                dup_pos = j
                break
        if dup_pos == -1:
            kept.append(cand)
            hashes.append(h)
        elif cand.fullness > kept[dup_pos].fullness:
            kept[dup_pos] = cand
            hashes[dup_pos] = h

    return sorted(kept, key=lambda c: c.time)
