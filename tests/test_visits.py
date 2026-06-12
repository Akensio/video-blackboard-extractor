import numpy as np

from vbe.boards import visits


# ---------------------------------------------------------------- max_pool

def test_max_pool_preserves_thin_strokes():
    m = np.zeros((40, 40), dtype=bool)
    m[10, :] = True                      # a 1-px chalk line
    out = visits.max_pool(m, 4)
    assert out.shape == (10, 10)
    assert out[2].all()                  # the line survives pooling


def test_max_pool_k1_identity():
    m = np.random.default_rng(0).random((8, 8)) > 0.5
    assert np.array_equal(visits.max_pool(m, 1), m)


# ---------------------------------------------------------- segment_visits

def _presence(spans, n=600):
    p = np.zeros(n, dtype=bool)
    for a, b in spans:
        p[a:b] = True
    return p


def test_segment_visits_basic():
    p = _presence([(100, 200), (400, 480)])
    out = visits.segment_visits(p, fps=1.0, min_seconds=8, bridge_seconds=12)
    assert [(v.start, v.end) for v in out] == [(100, 199), (400, 479)]


def test_segment_visits_bridges_short_gaps():
    # 10s gap < 12s bridge -> one visit
    p = _presence([(100, 150), (160, 220)])
    out = visits.segment_visits(p, fps=1.0, min_seconds=8, bridge_seconds=12)
    assert [(v.start, v.end) for v in out] == [(100, 219)]


def test_segment_visits_keeps_long_gaps_separate():
    p = _presence([(100, 150), (200, 260)])  # 50s gap > bridge
    out = visits.segment_visits(p, fps=1.0, min_seconds=8, bridge_seconds=12)
    assert len(out) == 2


def test_segment_visits_drops_walk_bys():
    p = _presence([(100, 104)])  # 4s < 8s minimum
    assert visits.segment_visits(p, fps=1.0, min_seconds=8, bridge_seconds=12) == []


# -------------------------------------------------------- change_fraction

def _state(seed=0, density=0.1, shape=(80, 60)):
    rng = np.random.default_rng(seed)
    return rng.random(shape) < density


def test_change_identical_is_zero():
    s = _state()
    assert visits.change_fraction(s, s.copy(), max_shift=40) == 0.0


def test_change_detects_added_chalk():
    ref = _state()
    cur = ref.copy()
    cur[40:50, 10:50] = True             # a new written block
    frac = visits.change_fraction(cur, ref, max_shift=40)
    assert frac > 0.05


def test_change_detects_removed_chalk():
    ref = _state(density=0.2)
    cur = ref.copy()
    cur[10:30, :] = False                # partial erase
    assert visits.change_fraction(cur, ref, max_shift=40) > 0.02


def test_pure_slide_is_not_a_change():
    ref = _state(density=0.15)
    cur = np.roll(ref, 14, axis=0)       # board slid down 14 cells
    raw = visits.change_fraction(cur, ref, max_shift=0)
    comp = visits.change_fraction(cur, ref, max_shift=40)
    assert raw > 0.1                     # naive diff sees a big change
    assert comp < 0.02                   # slide-compensated diff does not


def test_slide_plus_writing_reports_only_the_writing():
    ref = _state(density=0.15)
    cur = np.roll(ref, 14, axis=0)
    cur[60:70, 5:55] = True              # new chalk after the slide
    comp = visits.change_fraction(cur, ref, max_shift=40)
    assert 0.01 < comp < 0.2


def test_change_shape_mismatch_is_full_change():
    assert visits.change_fraction(np.zeros((4, 4), bool), np.zeros((5, 5), bool)) >= 1.0


def test_added_removed_distinguishes_writing_from_wipe():
    ref = _state(density=0.15)
    written = ref.copy()
    written[40:60, 10:50] = True
    add, rem = visits.added_removed(written, ref, max_shift=40)
    assert add > 0.05 and rem < 0.01     # writing: add-dominant

    wiped = ref.copy()
    wiped[:40, :] = False
    add, rem = visits.added_removed(wiped, ref, max_shift=40)
    assert rem > 0.02 and add < 0.005    # wipe: remove-dominant


def test_added_removed_symmetric_drift():
    rng = np.random.default_rng(3)
    ref = _state(density=0.15)
    cur = ref.copy()
    flip = rng.random(ref.shape) < 0.004   # smear-like symmetric flicker
    cur[flip] = ~cur[flip]
    add, rem = visits.added_removed(cur, ref, max_shift=40)
    assert abs(add - rem) < 0.004          # roughly symmetric, both small


# -------------------------------------------------------- writing_window

def test_writing_window_brackets_the_rise():
    # flat, then content appears over indices 100-200, then flat again
    curve = np.concatenate([
        np.zeros(100), np.linspace(0, 0.05, 100), np.full(100, 0.05)])
    ws, we = visits.writing_window(curve, floor=0.003)
    assert 95 <= ws <= 115           # start when change becomes non-trivial
    assert 175 <= we <= 205          # end when ~all of it has appeared


def test_writing_window_empty_curve():
    assert visits.writing_window(np.array([]), floor=0.01) == (0, 0)


def test_writing_window_monotone_orders():
    curve = np.linspace(0, 0.03, 50)
    ws, we = visits.writing_window(curve, floor=0.001)
    assert 0 <= ws <= we <= 49
