# video-blackboard-extractor (`vbe`)

Extract a **per-board writing timeline** (clean, legible board snapshots with the time
intervals during which they were written) and a **timestamped transcript** from
static-camera lecture recordings. Built for ~2-hour physics lectures filmed with a fixed
wide shot of a wall of sliding chalk blackboards, where a lecturer periodically walks in
front of the board.

The output is designed as a ready-to-walk basis for LLM lecture-note generation:

1. **Board timeline** - the board wall is three columns of sliding boards. Per column,
   the lecturer's *visits* (sustained presence, from the occlusion signal) and *erase
   events* drive snapshot capture: when he walks away from a column, when a wipe is about
   to start, or after a long quiet stretch, the column's chalk state is diffed against the
   previous snapshot (slide-compensated, so boards merely sliding don't count) and a
   snapshot is emitted only if content actually changed. Each snapshot is the cleanest
   post-trigger moment, exported with the lecturer removed, as a full-column crop + a
   CLAHE-enhanced legibility variant + a full-wall context frame. So a half-written board
   during `[a,b]`, the fuller board during `[c,d]`, and a fresh `board_id` after each erase.
2. **Transcript** - local `faster-whisper` (large-v3) transcription with VAD, word-level
   timestamps and a physics glossary, exported as `srt` / `vtt` / `txt` / `json`.

`timeline.json` ties it together: chronological snapshots, each with its
`writing_interval`, board lifecycle (`board`, `final`, `erased_at`) and - after `vbe pair` -
the transcript text spoken while that content was being written and explained.

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
timeline.json                       the LLM-ready index (see below)
montage.png                         contact sheet of all snapshots
boards/<column>/bBB_sSS_tHH-MM-SS.png        tight board crop (lecturer removed)
boards/<column>/bBB_sSS_tHH-MM-SS_enh.png    CLAHE-enhanced legibility variant
wall/tHH-MM-SS.png                  full-wall context frame
transcript.srt | .vtt | .txt | .json
```

### timeline.json snapshot fields

| field | meaning |
|---|---|
| `column` | which board column (`left` / `center` / `right`) |
| `board` | board identity, e.g. `left#2` = the 2nd board on the left column (increments at each erase) |
| `writing_interval` | `[a, b]` seconds: when this content was written (also `_str` as `HH:MM:SS`) |
| `capture_time` | when the snapshot image was taken (just after writing stopped) |
| `final` | `true` if this is the board's most complete state (no later snapshot before erase/end) |
| `erased_at` | when this board was wiped (`null` = never within the analyzed range) |
| `image` / `image_enhanced` / `wall_image` | the exported images |
| `transcript` | `{start, end, text}` spoken from this burst's start until the next snapshot's burst (after `vbe pair`) |

Snapshots of the same `board` are progressive states of the same physical board filling up;
the `final: true` one is the most complete. See `configs/default.yaml` for all tunables.
