import numpy as np

from vbe.boards import bursts


def _signal(fps=1.0, pieces=()):
    """Build a piecewise-linear fullness signal from (duration_s, end_value) pieces."""
    vals = [0.0]
    for dur, end in pieces:
        n = int(dur * fps)
        seg = np.linspace(vals[-1], end, n + 1)[1:]
        vals.extend(seg.tolist())
    return np.array(vals)


def test_write_pause_write_two_bursts():
    # write 120s to 0.06, talk 200s flat, write 120s to 0.12  (gap > merge_gap)
    f = _signal(pieces=[(120, 0.06), (200, 0.06), (120, 0.12)])
    out = bursts.detect_bursts(f, fps=1.0, rate_threshold=0.00025,
                               merge_gap_seconds=60, min_gain=0.012)
    assert len(out) == 2
    a, b = out
    assert a.start < a.end < b.start < b.end
    assert a.gain > 0.012 and b.gain > 0.012


def test_short_pause_merges_into_one_burst():
    # 30s pause < merge_gap of 60 -> single burst spanning both writing stretches
    f = _signal(pieces=[(120, 0.06), (30, 0.06), (120, 0.12)])
    out = bursts.detect_bursts(f, fps=1.0, rate_threshold=0.00025,
                               merge_gap_seconds=60, min_gain=0.012)
    assert len(out) == 1
    assert out[0].gain > 0.1


def test_no_merge_across_erase():
    # write, brief pause containing an ERASE, write again -> two bursts
    f = _signal(pieces=[(120, 0.06), (10, 0.06), (5, 0.005), (10, 0.005), (120, 0.07)])
    erase_at = 132  # inside the drop
    out = bursts.detect_bursts(f, fps=1.0, rate_threshold=0.00025,
                               merge_gap_seconds=60, min_gain=0.012,
                               erase_indices=[erase_at])
    assert len(out) == 2


def test_tiny_gain_burst_dropped():
    f = _signal(pieces=[(60, 0.005), (300, 0.005)])  # adds only 0.005 fullness
    out = bursts.detect_bursts(f, fps=1.0, rate_threshold=0.00025,
                               merge_gap_seconds=60, min_gain=0.012)
    assert out == []


def test_flat_signal_no_bursts():
    f = np.full(600, 0.08)
    out = bursts.detect_bursts(f, fps=1.0, rate_threshold=0.00025,
                               merge_gap_seconds=60, min_gain=0.012)
    assert out == []


def test_fill_nan_interpolates():
    v = np.array([0.1, np.nan, np.nan, 0.4])
    out = bursts.fill_nan(v)
    assert np.allclose(out, [0.1, 0.2, 0.3, 0.4])


def test_fill_nan_all_nan_is_zeros():
    out = bursts.fill_nan(np.array([np.nan, np.nan]))
    assert np.allclose(out, [0.0, 0.0])
