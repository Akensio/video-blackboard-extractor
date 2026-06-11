"""Board-timeline extraction orchestrator.

Per column of boards (the camera wall has three), this builds the lecture's
writing timeline:

  decode low-fps frames -> occlusion-aware chalk-fullness per column
  -> erase events (board lifecycle) and writing bursts (write/talk/write)
  -> one snapshot per burst, captured at the cleanest post-burst moment
  -> full-res lecturer-removed export: column crop (+ enhanced) + wall context
  -> timeline.json: chronological snapshots with writing intervals.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..config import Config
from ..io import ffmpeg
from ..io.frame_cache import FrameCache
from . import background, bursts, enhance, epochs, fullness, keyframe, montage, person, roi


def _fmt_t(seconds: float) -> str:
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}-{m:02d}-{s:02d}"


def _clock(seconds: float) -> str:
    return _fmt_t(seconds).replace("-", ":")


@dataclass
class TimelineResult:
    video: str
    duration: float
    columns: list[str] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)
    montage: str | None = None
    analysis_start: float = 0.0
    analysis_end: float = 0.0


@dataclass
class _Snapshot:
    column: str
    board_index: int          # 1-based, increments at each erase in the column
    burst: bursts.Burst
    capture_index: int
    capture_time: float
    fullness: float
    occlusion: float
    erased_at: float | None = None
    final: bool = False


def _signals(cache: FrameCache, cfg: Config, columns, masks, bboxes, grid):
    """Single pass over cached frames -> per-column occlusion-aware signals.

    Fullness uses a per-pixel CHALK-HOLD estimator: each pixel keeps its last
    chalk reading from a frame where it was visible, so the measured fullness
    is always over the column's full area. Dividing by only the visible area
    would bias the estimate up or down depending on whether the lecturer covers
    a blank or a written part of the board, creating phantom writing bursts.
    """
    n = cache.n
    names = [c.name for c in columns]
    raw = {nm: np.zeros(n) for nm in names}
    occ = {nm: np.zeros(n) for nm in names}
    sharp = {nm: np.zeros(n) for nm in names}

    areas = {nm: int(masks[nm].sum()) for nm in names}
    chalk_hold = np.zeros((cache.height, cache.width), dtype=bool)
    for i in tqdm(range(n), desc="analysing frames", unit="f"):
        gray = cache.get(i)
        chalk = fullness.chalk_mask(gray, cfg.chalk_tophat_kernel, cfg.chalk_threshold)
        pmask = person.person_foreground(gray, grid.nearest(i), dilate_px=cfg.person_dilate_px)
        visible = ~pmask
        chalk_hold[visible] = chalk[visible]
        lap = None
        for nm in names:
            m = masks[nm]
            area = areas[nm]
            if area == 0:
                continue
            occ[nm][i] = 1.0 - int((m & visible).sum()) / area
            raw[nm][i] = np.count_nonzero(chalk_hold & m) / area
            # Sharpness over visible board pixels only - the lecturer's edges
            # otherwise dominate the Laplacian and reward occluded frames.
            x0, y0, x1, y1 = bboxes[nm]
            if lap is None:
                lap = cv2.Laplacian(gray, cv2.CV_64F)
            region_vis = visible[y0:y1, x0:x1]
            vals = lap[y0:y1, x0:x1][region_vis]
            sharp[nm][i] = float(vals.var()) if vals.size > 100 else 0.0

    smooth = {nm: fullness.smooth_signal(raw[nm], cfg.fullness_smooth_window) for nm in names}
    return smooth, occ, sharp


def _column_snapshots(nm, f, occ_sig, sharp_sig, cache, cfg, fps) -> list[_Snapshot]:
    """Erase events + writing bursts -> capture-scored snapshots for one column."""
    refractory = max(1, round(cfg.erase_refractory_seconds * fps))
    persist = max(1, round(cfg.erase_persist_seconds * fps))
    events = epochs.detect_events(
        f, cfg.erase_min_drop, refractory,
        min_peak=cfg.erase_min_peak, persist_frames=persist,
    )
    erase_idx = [e.drop_index for e in events]

    blist = bursts.detect_bursts(
        f, fps,
        rate_threshold=cfg.burst_rate_threshold,
        merge_gap_seconds=cfg.burst_merge_gap_seconds,
        min_gain=cfg.burst_min_gain,
        erase_indices=erase_idx,
        deriv_window_seconds=cfg.burst_deriv_window_seconds,
    )

    scores = keyframe.score_frames(
        f, occ_sig, sharp_sig,
        cfg.weight_fullness, cfg.weight_occlusion, cfg.weight_sharpness,
    )
    post = max(1, round(cfg.burst_post_window_seconds * fps))

    snaps: list[_Snapshot] = []
    for b in blist:
        hi = b.end + post + 1
        nxt_event = next((e for e in events if e.drop_index > b.end), None)
        if nxt_event is not None:
            # Cap at the pre-erase PEAK: frames between the peak and the
            # detected drop are already mid-wipe.
            hi = min(hi, max(nxt_event.peak_index + 1, b.end + 1))
        ci = keyframe.select_best_index(scores, b.end, max(hi, b.end + 1))
        board_index = 1 + sum(1 for e in erase_idx if e <= b.start)
        erased_at = (float(cache.times[nxt_event.drop_index])
                     if nxt_event is not None else None)
        snaps.append(_Snapshot(
            column=nm, board_index=board_index, burst=b,
            capture_index=ci, capture_time=float(cache.times[ci]),
            fullness=float(f[ci]), occlusion=float(occ_sig[ci]),
            erased_at=erased_at,
        ))

    # final = last snapshot of each board (next column event is an erase or video end)
    for i, s in enumerate(snaps):
        later_same_board = any(
            t.board_index == s.board_index for t in snaps[i + 1:]
        )
        s.final = not later_same_board
    return snaps


def _merge_near_duplicates(snaps: list[_Snapshot], cache: FrameCache, bboxes,
                           phash_max: int) -> list[_Snapshot]:
    """Within a column, merge consecutive snapshots whose board content is
    near-identical (a burst that added almost nothing visible). The later
    snapshot wins; its writing interval is extended back to the earlier start."""
    import imagehash
    from PIL import Image

    def crop_hash(s: _Snapshot):
        x0, y0, x1, y1 = bboxes[s.column]
        return imagehash.phash(Image.fromarray(cache.get(s.capture_index)[y0:y1, x0:x1]))

    out: list[_Snapshot] = []
    last_in_column: dict[str, int] = {}  # column -> index into `out`
    for s in sorted(snaps, key=lambda x: x.capture_time):
        j = last_in_column.get(s.column)
        if j is not None:
            prev = out[j]
            if (prev.board_index == s.board_index
                    and (crop_hash(prev) - crop_hash(s)) <= phash_max):
                merged_burst = bursts.Burst(
                    start=prev.burst.start, end=s.burst.end,
                    gain=prev.burst.gain + s.burst.gain,
                )
                out[j] = _Snapshot(
                    column=s.column, board_index=s.board_index, burst=merged_burst,
                    capture_index=s.capture_index, capture_time=s.capture_time,
                    fullness=s.fullness, occlusion=s.occlusion,
                    erased_at=s.erased_at, final=s.final or prev.final,
                )
                continue
        out.append(s)
        last_in_column[s.column] = len(out) - 1
    return out


def _global_color_plate(video_path, times: np.ndarray, k: int = 41) -> np.ndarray | None:
    """Full-resolution, person-free color plate over the whole analysis range."""
    t0, t1 = float(times[0]), float(times[-1])
    span = max(t1 - t0, 1.0)
    fps_g = min(1.0, k / span)
    frames = [f for _, f in ffmpeg.decode_frames(
        video_path, fps=fps_g, start=t0, duration=span, size=None, gray=False)]
    if not frames:
        return None
    return background.median_plate(np.stack(frames, axis=0))


def run_boards(
    video_path: str | Path,
    cfg: Config,
    out_dir: str | Path,
    start: float | None = None,
    duration: float | None = None,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
) -> TimelineResult:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_root) if cache_root else out_dir / "_cache"

    info = ffmpeg.probe(video_path)
    fps = cfg.analysis_fps

    cache = FrameCache.load_or_build(
        video_path, info, fps=fps, analysis_width=cfg.analysis_width,
        cache_root=cache_root, start=start, duration=duration, rebuild=rebuild,
    )
    if cache.n == 0:
        return TimelineResult(video=video_path.name, duration=info.duration)

    scale = cache.width / info.width
    columns = roi.panel_rois(cfg.rois)
    if not columns:
        columns = [roi.default_full_roi(info.width, info.height)]
    masks = {c.name: roi.roi_mask(c, cache.height, cache.width, scale) for c in columns}
    bboxes = {c.name: roi.scaled_bbox(c, scale, cache.width, cache.height) for c in columns}
    full_bboxes = {c.name: roi.fullres_bbox(c, info.width, info.height) for c in columns}

    half = max(1, round(cfg.median_window_seconds * fps))
    grid = background.build_plate_grid(cache, half_window_frames=half)

    smooth, occ, sharp = _signals(cache, cfg, columns, masks, bboxes, grid)

    snaps: list[_Snapshot] = []
    for c in columns:
        nm = c.name
        snaps.extend(_column_snapshots(nm, smooth[nm], occ[nm], sharp[nm], cache, cfg, fps))
    snaps = _merge_near_duplicates(snaps, cache, bboxes, cfg.phash_max_distance)
    snaps.sort(key=lambda s: s.capture_time)

    # --- full-resolution export ---
    boards_dir = out_dir / "boards"
    wall_dir = out_dir / "wall"
    for d in (boards_dir, wall_dir):
        if d.exists():
            shutil.rmtree(d)  # avoid mixing stale snapshots from prior runs
        d.mkdir(parents=True)

    segmenter = None
    if cfg.person_removal == "gated_median":
        try:
            segmenter = person.YoloPersonSegmenter(cfg.seg_model)
        except Exception as exc:
            print(f"[boards] gated_median requested but segmenter unavailable ({exc}); "
                  f"using median compositing.")

    global_plate = _global_color_plate(video_path, cache.times)

    clean_cache: dict[int, np.ndarray] = {}
    records: list[dict] = []
    montage_imgs: list[Path] = []
    montage_caps: list[str] = []
    counters: dict[str, int] = {}

    for sid, s in enumerate(tqdm(snaps, desc="exporting snapshots", unit="snap"), start=1):
        if s.capture_index not in clean_cache:
            clean_cache[s.capture_index] = keyframe.build_clean_frame(
                video_path, s.capture_time, cfg.export_window_seconds, cfg.export_fps,
                global_plate=global_plate,
                dilate_px=cfg.person_dilate_px + 2, segmenter=segmenter,
            )
        clean = clean_cache[s.capture_index]

        tstamp = _fmt_t(s.capture_time)
        wall_path = wall_dir / f"t{tstamp}.{cfg.export_format}"
        if not wall_path.exists():
            cv2.imwrite(str(wall_path), clean)

        counters[s.column] = counters.get(s.column, 0) + 1
        col_dir = boards_dir / s.column
        col_dir.mkdir(exist_ok=True)
        base = f"b{s.board_index:02d}_s{counters[s.column]:02d}_t{tstamp}"
        x0, y0, x1, y1 = full_bboxes[s.column]
        crop = enhance.trim_to_board(clean[y0:y1, x0:x1])
        crop_path = col_dir / f"{base}.{cfg.export_format}"
        cv2.imwrite(str(crop_path), crop)

        enh_rel = None
        if cfg.enhance_crops:
            enh = enhance.enhance_board_crop(crop)
            enh_path = col_dir / f"{base}_enh.{cfg.export_format}"
            cv2.imwrite(str(enh_path), enh)
            enh_rel = enh_path.relative_to(out_dir).as_posix()

        t_a = float(cache.times[s.burst.start])
        t_b = float(cache.times[s.burst.end])
        records.append({
            "id": sid,
            "column": s.column,
            "board": f"{s.column}#{s.board_index}",
            "writing_interval": [round(t_a, 1), round(t_b, 1)],
            "writing_interval_str": [_clock(t_a), _clock(t_b)],
            "capture_time": round(s.capture_time, 1),
            "capture_time_str": _clock(s.capture_time),
            "final": s.final,
            "final_reason": (None if not s.final
                             else ("erased" if s.erased_at is not None
                                   else "end_of_analysis")),
            "erased_at": round(s.erased_at, 1) if s.erased_at is not None else None,
            "fullness": round(s.fullness, 4),
            "occlusion": round(s.occlusion, 4),
            "image": crop_path.relative_to(out_dir).as_posix(),
            "image_size": [crop.shape[1], crop.shape[0]],
            "image_enhanced": enh_rel,
            "wall_image": wall_path.relative_to(out_dir).as_posix(),
        })
        montage_imgs.append(crop_path)
        montage_caps.append(
            f"#{sid} {s.column} b{s.board_index} [{_clock(t_a)}-{_clock(t_b)}]"
            + (" FINAL" if s.final else "")
        )

    montage_rel = None
    if montage_imgs:
        mp = montage.make_montage(montage_imgs, montage_caps, out_dir / "montage.png")
        montage_rel = str(mp.relative_to(out_dir))

    return TimelineResult(
        video=video_path.name,
        duration=info.duration,
        columns=[c.name for c in columns],
        snapshots=records,
        montage=montage_rel,
        analysis_start=float(cache.times[0]),
        analysis_end=float(cache.times[-1]),
    )
