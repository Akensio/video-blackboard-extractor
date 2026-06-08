import numpy as np

from vbe.boards import fullness


def test_chalk_mask_detects_thin_bright_strokes():
    img = np.full((100, 100), 20, dtype=np.uint8)  # dark board
    img[40:42, 10:90] = 230   # a thin bright chalk line
    img[10:90, 60:62] = 230   # another stroke
    mask = fullness.chalk_mask(img, tophat_kernel=15, threshold=18)
    assert mask[41, 50]       # on a stroke
    assert not mask[5, 5]     # empty board


def test_chalk_mask_ignores_large_uniform_block():
    img = np.full((100, 100), 20, dtype=np.uint8)
    img[20:80, 20:80] = 200   # large bright block (like a body), not a thin stroke
    mask = fullness.chalk_mask(img, tophat_kernel=15, threshold=18)
    # interior of a large block is suppressed by top-hat
    assert not mask[50, 50]


def test_fullness_in_mask_monotonic():
    region = np.ones((100, 100), dtype=bool)
    empty = np.full((100, 100), 20, dtype=np.uint8)
    written = empty.copy()
    for y in range(10, 90, 6):
        written[y:y + 1, 10:90] = 230
    f_empty = fullness.fullness_in_mask(empty, region, 15, 18)
    f_written = fullness.fullness_in_mask(written, region, 15, 18)
    assert f_written > f_empty


def test_smooth_signal_preserves_length():
    v = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    out = fullness.smooth_signal(v, window=5)
    assert len(out) == len(v)


def test_laplacian_sharpness_blur_is_lower():
    import cv2
    rng = np.random.default_rng(0)
    sharp = (rng.random((80, 80)) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(sharp, (9, 9), 4)
    bbox = (0, 0, 80, 80)
    assert fullness.laplacian_sharpness(sharp, bbox) > fullness.laplacian_sharpness(blur, bbox)
