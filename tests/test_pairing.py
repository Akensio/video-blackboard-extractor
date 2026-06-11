import json

from vbe.pairing import pair_timeline


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_pair_attaches_burst_window_text(tmp_path):
    transcript = tmp_path / "transcript.json"
    timeline = tmp_path / "timeline.json"
    _write(transcript, {
        "segments": [
            {"start": 0, "end": 5, "text": "intro"},
            {"start": 100, "end": 130, "text": "writing board one"},
            {"start": 200, "end": 240, "text": "explaining board one"},
            {"start": 400, "end": 450, "text": "writing board two"},
        ]
    })
    _write(timeline, {
        "video": "v.mp4",
        "snapshots": [
            {"id": 1, "column": "left", "writing_interval": [95, 180],
             "capture_time": 190},
            {"id": 2, "column": "center", "writing_interval": [395, 460],
             "capture_time": 470},
        ],
    })

    pair_timeline(timeline, transcript, margin_seconds=10)
    out = json.loads(timeline.read_text(encoding="utf-8"))
    s1, s2 = out["snapshots"]

    # snapshot 1 window [85, 405): the writing AND the explanation that follows,
    # but not board two's writing text... margin makes the boundary 405 so the
    # segment starting at 400 IS included at the edge - that overlap is by design.
    assert "writing board one" in s1["transcript"]["text"]
    assert "explaining board one" in s1["transcript"]["text"]
    assert "intro" not in s1["transcript"]["text"]
    # snapshot 2 (last) takes everything from its burst start onward
    assert "writing board two" in s2["transcript"]["text"]
    assert "explaining board one" not in s2["transcript"]["text"]
    assert out["transcript_file"] == "transcript.json"


def test_pair_empty_window_gives_empty_text(tmp_path):
    transcript = tmp_path / "transcript.json"
    timeline = tmp_path / "timeline.json"
    _write(transcript, {"segments": [{"start": 0, "end": 5, "text": "intro"}]})
    _write(timeline, {
        "video": "v.mp4",
        "snapshots": [{"id": 1, "column": "left",
                       "writing_interval": [500, 600], "capture_time": 610}],
    })
    pair_timeline(timeline, transcript, margin_seconds=5)
    out = json.loads(timeline.read_text(encoding="utf-8"))
    assert out["snapshots"][0]["transcript"]["text"] == ""
    assert out["snapshots"][0]["transcript"]["start"] is None
