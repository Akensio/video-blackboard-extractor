import numpy as np

from vbe.boards import epochs


def test_detect_events_finds_drops():
    # ramps up to 0.5, erased to 0.1, ramps to 0.4, erased to 0.05
    f = np.array([0.0, 0.1, 0.3, 0.5, 0.5, 0.5, 0.1, 0.1, 0.2, 0.4, 0.4, 0.05])
    events = epochs.detect_events(f, min_drop=0.2, refractory_frames=1)
    assert len(events) == 2
    # first event's peak is at one of the 0.5 plateau indices (3,4,5); drop after
    assert f[events[0].peak_index] == 0.5
    assert events[0].drop_index >= 6
    assert f[events[1].peak_index] == 0.4


def test_detect_events_respects_min_drop():
    f = np.array([0.0, 0.1, 0.2, 0.18, 0.2, 0.19])  # only tiny dips
    assert epochs.detect_events(f, min_drop=0.2, refractory_frames=1) == []


def test_capture_indices_includes_peaks_and_tail():
    f = np.array([0.0, 0.2, 0.5, 0.1, 0.3, 0.6, 0.6])  # one drop, then rising tail
    events = epochs.detect_events(f, min_drop=0.2, refractory_frames=1)
    caps = epochs.capture_indices(f, events)
    assert 2 in caps              # the pre-erase peak
    assert max(caps) >= 5         # the trailing (final) state


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
