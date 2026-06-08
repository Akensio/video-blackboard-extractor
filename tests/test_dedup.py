import cv2
import numpy as np

from vbe.boards import dedup, keyframe


def _structured_image(seed: int = 0) -> np.ndarray:
    """A blocky textured image with rich high-frequency content (stable pHash),
    unlike a smooth gradient which makes pHash degenerate."""
    rng = np.random.default_rng(seed)
    small = (rng.random((16, 16)) * 255).astype(np.uint8)
    big = cv2.resize(small, (200, 200), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


def test_deduplicate_merges_near_duplicates_keeping_fuller():
    a = _structured_image(0)
    rng = np.random.default_rng(99)
    a_noisy = np.clip(a.astype(int) + rng.integers(-4, 5, a.shape), 0, 255).astype(np.uint8)
    b = _structured_image(7)  # clearly different

    cands = [
        dedup.KeyframeCandidate(index=0, time=0.0, image=a, fullness=0.4, occlusion=0.0, meta={}),
        dedup.KeyframeCandidate(index=1, time=10.0, image=a_noisy, fullness=0.7, occlusion=0.0, meta={}),
        dedup.KeyframeCandidate(index=2, time=20.0, image=b, fullness=0.5, occlusion=0.0, meta={}),
    ]
    kept = dedup.deduplicate(cands, phash_max_distance=8, ssim_tiebreak=0.92)
    assert len(kept) == 2
    # the near-duplicate group keeps the fuller one (fullness 0.7)
    fullnesses = sorted(k.fullness for k in kept)
    assert fullnesses == [0.5, 0.7]


def test_deduplicate_keeps_distinct():
    a = _structured_image(0)
    b = _structured_image(7)
    cands = [
        dedup.KeyframeCandidate(index=0, time=0.0, image=a, fullness=0.4, occlusion=0.0, meta={}),
        dedup.KeyframeCandidate(index=1, time=5.0, image=b, fullness=0.4, occlusion=0.0, meta={}),
    ]
    kept = dedup.deduplicate(cands, phash_max_distance=8, ssim_tiebreak=0.92)
    assert len(kept) == 2


def test_score_and_select():
    fulln = np.array([0.1, 0.5, 0.9, 0.9])
    occl = np.array([0.0, 0.0, 0.8, 0.0])   # index 2 heavily occluded
    sharp = np.array([1.0, 1.0, 1.0, 1.0])
    scores = keyframe.score_frames(fulln, occl, sharp, 1.0, 1.5, 0.4)
    best = keyframe.select_best_index(scores, 0, 4)
    assert best == 3   # full but unoccluded beats full-but-occluded
