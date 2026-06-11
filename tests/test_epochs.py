import numpy as np

from vbe.boards import epochs


def test_detect_events_finds_relative_drops():
    # board fills to 0.15 (a realistic full-board fullness), then is wiped to
    # 0.03 (80% relative drop) and refills to 0.10, wiped again to 0.01.
    f = np.concatenate([
        np.linspace(0.0, 0.15, 60),   # write
        np.full(30, 0.15),            # talk
        np.full(20, 0.03),            # ERASED (drop persists)
        np.linspace(0.03, 0.10, 50),  # write again
        np.full(20, 0.01),            # ERASED again
    ])
    events = epochs.detect_events(f, min_drop=0.35, refractory_frames=5,
                                  min_peak=0.03, persist_frames=8)
    assert len(events) == 2
    assert abs(f[events[0].peak_index] - 0.15) < 1e-9
    assert 89 <= events[0].drop_index <= 95
    assert abs(f[events[1].peak_index] - 0.10) < 1e-6


def test_detect_events_ignores_transient_dips():
    # a 4-frame occlusion dip must NOT count as an erase (persistence = 8)
    f = np.full(100, 0.15)
    f[50:54] = 0.05
    events = epochs.detect_events(f, min_drop=0.35, refractory_frames=5,
                                  min_peak=0.03, persist_frames=8)
    assert events == []


def test_detect_events_ignores_near_empty_boards():
    # tiny signal on an almost-empty board: relative drop is large but the
    # peak is below min_peak, so no event
    f = np.concatenate([np.full(30, 0.02), np.full(30, 0.005)])
    events = epochs.detect_events(f, min_drop=0.35, refractory_frames=5,
                                  min_peak=0.03, persist_frames=8)
    assert events == []


def test_detect_events_small_relative_drop_no_event():
    # 0.15 -> 0.12 is only a 20% drop: below the 35% threshold
    f = np.concatenate([np.full(30, 0.15), np.full(30, 0.12)])
    events = epochs.detect_events(f, min_drop=0.35, refractory_frames=5,
                                  min_peak=0.03, persist_frames=8)
    assert events == []


def test_capture_indices_includes_peaks_and_tail():
    f = np.concatenate([
        np.linspace(0.0, 0.12, 40),
        np.full(15, 0.02),            # erase (persists)
        np.linspace(0.02, 0.08, 40),  # rising tail
    ])
    events = epochs.detect_events(f, min_drop=0.35, refractory_frames=5,
                                  min_peak=0.03, persist_frames=8)
    caps = epochs.capture_indices(f, events)
    assert events and events[0].peak_index in caps   # the pre-erase peak
    assert max(caps) > 60                            # the trailing (final) state


def test_classify_slide_detects_vertical_shift():
    rng = np.random.default_rng(0)
    patch = (rng.random((64, 64)) * 255).astype(np.uint8)
    shifted = np.roll(patch, 6, axis=0)
    is_slide, dy = epochs.classify_slide(patch, shifted, shift_threshold_px=3)
    assert is_slide
    assert abs(abs(dy) - 6) <= 1.5


def test_classify_slide_negative_on_static():
    rng = np.random.default_rng(1)
    patch = (rng.random((64, 64)) * 255).astype(np.uint8)
    is_slide, dy = epochs.classify_slide(patch, patch.copy(), shift_threshold_px=3)
    assert not is_slide
    assert abs(dy) < 1.0
