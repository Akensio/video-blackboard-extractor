# video-blackboard-extractor (`vbe`)

Extract **single-board snapshots** (clean, legible photos of each board the lecturer
finishes, with the time interval during which it was written) and a **timestamped
transcript** from static-camera lecture recordings. Built for ~2-hour physics lectures
filmed with a fixed wide shot of a wall of sliding chalk blackboards, where a lecturer
periodically walks in front of the board.

The output is designed as a ready-to-walk basis for LLM lecture-note generation:

1. **Board snapshots** - mimics a student photographing a board once the lecturer finishes
   it. The wall is watched as three columns; the lecturer's *visits* (sustained presence,
   from a YOLO person mask - the projector screen is excluded) are tracked, and **when he
   walks away from a board he just wrote on**, the cleanest now-unoccluded frame is captured,
   the lecturer is removed, the column is split at its wooden rail, and **each board that got
   fresh writing is exported as its own single-board image** (+ a CLAHE-enhanced legibility
   variant), plus a full-wall context frame. The rail is found per-frame, so vertical
   sliding of the boards never matters. No erase/lifecycle bookkeeping: if a board is later
   filled more, or wiped and rewritten, that simply produces another finished-board photo.
2. **Transcript** - local `faster-whisper` (large-v3) transcription with VAD, word-level
   timestamps and a physics glossary, exported as `srt` / `vtt` / `txt` / `json`.

`timeline.json` ties it together: chronological one-board-each snapshots, each with its
`column`/`vertical`/`location`, `writing_interval`, `capture_time` and - after `vbe pair` -
the transcript text spoken while that board was being written and explained.

## Safety

The lecture videos live in `gitignore/` and must **never** be committed. Derived outputs go
to a separate `output/` folder. Both are listed in `.gitignore`; the tool only ever writes
under `output/` (and the OS temp dir).

## Install

```powershell
uv venv --python 3.12
uv pip install -e .            # core (CPU transcription works out of the box)
uv pip install -e ".[seg]"     # + YOLO person segmentation (recommended, best quality)
uv pip install -e ".[gpu]"     # + CUDA libs (only useful for models smaller than large-v3)
```

Requires `ffmpeg`/`ffprobe` on `PATH`.

## Usage

`configs/wis_qft_2026.yaml` holds the board-column ROIs shared by all four 2026 QFT
lectures (same hall and camera). For a new hall, seed ROIs with `vbe boards-roi` and adjust.

```powershell
# Board timeline (snapshots + crops + montage + timeline.json)
vbe boards gitignore\qft_lec_2026_05_18_00.mp4 --config configs\wis_qft_2026.yaml --out output

# Transcript
vbe transcribe gitignore\qft_lec_2026_05_18_00.mp4 --config configs\wis_qft_2026.yaml --out output

# Attach per-snapshot transcript text to timeline.json
vbe pair gitignore\qft_lec_2026_05_18_00.mp4 --out output

# Or everything at once
vbe all gitignore\qft_lec_2026_05_18_00.mp4 --config configs\wis_qft_2026.yaml --out output

# Quick validation on a slice
vbe boards gitignore\qft_lec_2026_05_18_00.mp4 --config configs\wis_qft_2026.yaml `
    --start 1500 --duration 1800 --out output
```

## Outputs (per video, under `output/<video_name>/`)

```
timeline.json                          the LLM-ready index (see below)
montage.png                            contact sheet of all board snapshots
boards/<column>/<vert>_NN_tHH-MM-SS.png      single-board crop (lecturer removed)
boards/<column>/<vert>_NN_tHH-MM-SS_enh.png  CLAHE-enhanced legibility variant
wall/tHH-MM-SS.png                     full-wall context frame
transcript.srt | .vtt | .txt | .json
```

### timeline.json snapshot fields (each entry is ONE finished board)

| field | meaning |
|---|---|
| `column` | which wall column the board is in (`left` / `center` / `right`) |
| `vertical` | `upper` or `lower` board within that column at capture |
| `location` | `<column>_<vertical>`, a human label |
| `writing_interval` | `[a, b]` seconds: when this board was written (also `_str` as `HH:MM:SS`) |
| `capture_time` | when the photo was taken (just after he finished and stepped away) |
| `occlusion` | fraction of the board still hidden by the lecturer at capture |
| `image` / `image_enhanced` / `wall_image` | the exported images; `image_size` = `[w, h]` |
| `transcript` | `{start, end, text}` spoken while the board was written (after `vbe pair`) |

The same board can appear more than once (filled further, or wiped and rewritten); each is a
separate finished-board photo, in time order. See `configs/default.yaml` for all tunables.
