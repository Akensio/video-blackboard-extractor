import json

from vbe.pairing import pair_manifest


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_pair_attaches_text_window(tmp_path):
    transcript = tmp_path / "transcript.json"
    manifest = tmp_path / "manifest.json"
    _write(transcript, {
        "segments": [
            {"start": 0, "end": 5, "text": "intro"},
            {"start": 10, "end": 20, "text": "about board one"},
            {"start": 30, "end": 40, "text": "about board two"},
        ]
    })
    _write(manifest, {
        "video": "v.mp4",
        "keyframes": [
            {"id": 1, "time": 22, "image": "a.png"},
            {"id": 2, "time": 42, "image": "b.png"},
        ],
    })

    pair_manifest(manifest, transcript, margin_seconds=2)
    out = json.loads(manifest.read_text(encoding="utf-8"))
    kf1, kf2 = out["keyframes"]

    # kf1 window [0, 24] picks up intro + board one, not board two.
    assert "board one" in kf1["transcript_text"]
    assert "board two" not in kf1["transcript_text"]
    # kf2 window [22, 44] picks up board two.
    assert "board two" in kf2["transcript_text"]
    assert kf2["transcript_start"] == 30
