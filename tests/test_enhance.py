import numpy as np

from vbe.boards import enhance


def _bgr(gray2d):
    return np.repeat(np.asarray(gray2d, dtype=np.uint8)[..., None], 3, axis=2)


def test_trim_removes_bright_top_band():
    img = np.full((200, 100), 70, dtype=np.uint8)   # dark board
    img[:60, :] = 150                                # bright wood on top
    out = enhance.trim_to_board(_bgr(img), pad=5)
    # top band trimmed (minus pad), board retained
    assert out.shape[0] <= 200 - 60 + 5 + 1
    assert out.shape[0] >= 130


def test_trim_bridges_board_seam():
    img = np.full((300, 100), 70, dtype=np.uint8)    # two boards...
    img[140:148, :] = 160                            # ...with a thin bright seam
    out = enhance.trim_to_board(_bgr(img), pad=5)
    assert out.shape[0] == 300                       # seam bridged, nothing trimmed


def test_trim_keeps_glared_board_rows():
    img = np.full((200, 100), 70, dtype=np.uint8)
    img[:50, :] = 150                                # wood
    # glare: top board rows where 60% of pixels brighten but p25 stays dark
    img[50:90, 40:] = 140
    out = enhance.trim_to_board(_bgr(img), pad=0)
    assert out.shape[0] >= 150                       # glared rows survive


def test_trim_refuses_pathological():
    img = np.full((100, 100), 160, dtype=np.uint8)   # all wood, no board
    out = enhance.trim_to_board(_bgr(img))
    assert out.shape[:2] == (100, 100)               # unchanged


def test_enhance_returns_grayscale_same_size():
    img = _bgr(np.random.default_rng(0).integers(0, 255, (120, 80)))
    out = enhance.enhance_board_crop(img)
    assert out.shape == (120, 80)
    assert out.dtype == np.uint8
