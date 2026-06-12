import numpy as np

from vbe.boards import person


class _FakeCache:
    def __init__(self, n=20, h=12, w=16):
        self.n = n
        self._h, self._w = h, w

    def get(self, i):
        return np.full((self._h, self._w), i, dtype=np.uint8)


class _FakeSegmenter:
    """Marks one column per call; records which frames were segmented."""

    def __init__(self):
        self.calls = []

    def mask(self, frame_bgr, dilate_px=9):
        i = int(frame_bgr[0, 0, 0])
        self.calls.append(i)
        m = np.zeros(frame_bgr.shape[:2], dtype=bool)
        m[:, i % frame_bgr.shape[1]] = True
        return m


def test_sampled_mask_brackets_and_caches():
    cache = _FakeCache(n=20)
    seg = _FakeSegmenter()
    spm = person.SampledPersonMask(cache, seg, stride=5, dilate_px=0)

    m = spm.mask(7)             # slot 1 (frame 5) + slot 2 (frame 10)
    assert m[:, 5].all() and m[:, 10].all()   # union of both bracketing samples
    assert sorted(seg.calls) == [5, 10]

    spm.mask(8)                 # same slots -> served from cache
    assert sorted(seg.calls) == [5, 10]

    spm.mask(17)                # slot 3 (frame 15); slot 4 would start at 20 >= n
    assert 15 in seg.calls
    assert 19 not in seg.calls


def test_sampled_mask_clamps_last_sample():
    cache = _FakeCache(n=12)
    seg = _FakeSegmenter()
    spm = person.SampledPersonMask(cache, seg, stride=5, dilate_px=0)
    spm.mask(11)                # slot 2 -> frame min(10, 11)=10; slot 3 starts at 15 >= n
    assert seg.calls == [10]
