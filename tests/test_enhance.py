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


def test_board_cell_extent_snaps_to_rails_around_writing():
    # two dark boards (rows 20-90 and 130-200) split by a bright rail (90-130),
    # wood frame top (0-20) and bottom (200-220). Writing is on the lower board.
    col = np.full((220, 120), 70, dtype=np.uint8)   # dark board
    col[:20] = 150                                   # top wood
    col[90:130] = 150                                # mid rail
    col[200:] = 150                                  # bottom wood
    top, bot = enhance.board_cell_extent(col, y_lo=150, y_hi=180, pad=4)
    assert 126 <= top <= 134        # snaps up to the rail (130), minus pad
    assert 200 <= bot <= 206        # snaps down to the bottom wood (200)
    # and the UPPER band selects the upper board, not the lower
    top2, bot2 = enhance.board_cell_extent(col, y_lo=40, y_hi=70, pad=4)
    assert top2 < 30 and 86 <= bot2 <= 96


def test_board_cell_extent_fallback_when_no_board():
    col = np.full((100, 80), 160, dtype=np.uint8)   # all wood, no dark run
    top, bot = enhance.board_cell_extent(col, y_lo=40, y_hi=60, pad=5)
    assert top <= 40 and bot >= 60                  # falls back to padded band


def test_enhance_returns_grayscale_same_size():
    img = _bgr(np.random.default_rng(0).integers(0, 255, (120, 80)))
    out = enhance.enhance_board_crop(img)
    assert out.shape == (120, 80)
    assert out.dtype == np.uint8
