"""Board-snapshot extraction orchestrator.

Mimics a student photographing a board when the lecturer finishes it:

  decode low-fps frames -> per-column chalk-hold state + person/screen occlusion
  -> per column, detect the lecturer's VISITS (sustained presence, YOLO-backed)
     and capture when he WALKS AWAY from a board he just changed
  -> capture the cleanest frame once he is gone, crop the SINGLE board the fresh
     chalk is on (rail-to-rail, lecturer removed) -> board crop (+ enhanced) +
     wall context
  -> timeline.json: chronological one-board-each snapshots with writing
     intervals. No erase/lifecycle bookkeeping; a wipe-and-rewrite just yields
     another finished board.
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
from . import background, enhance, fullness, keyframe, montage, person, roi, visits


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
    column: str               # analysis zone the board sits in (left/center/right)
    write_start_i: int        # analysis-frame index: content began appearing
    write_end_i: int          # analysis-frame index: content was complete
    capture_index: int        # analysis-frame index the photo is taken at
    capture_time: float
    occlusion: float
    added_mask: np.ndarray    # pooled state grid: clustered cells of fresh chalk
    visit_spans: list[tuple[int, int]] = field(default_factory=list)


def _signals(cache: FrameCache, cfg: Config, columns, masks, bboxes, grid):
    """Single pass over cached frames -> per-column signals + chalk states.

    Fullness uses a per-pixel CHALK-HOLD estimator: each pixel keeps its last
    chalk reading from a frame where it was visible, so the measured fullness
    is always over the column's full area and does not jump when the lecturer
    covers blank vs written board.

    Also stores, per column and frame, a max-pooled snapshot of the chalk-hold
    state - the basis for the "did this column actually change" diffs that
    gate snapshot emission.
    """
    n = cache.n
    names = [c.name for c in columns]
    raw = {nm: np.zeros(n) for nm in names}
    occ = {nm: np.zeros(n) for nm in names}        # lecturer OR screen hides the board
    person_occ = {nm: np.zeros(n) for nm in names}  # lecturer only -> presence/walk-away
    sharp = {nm: np.zeros(n) for nm in names}

    areas = {nm: int(masks[nm].sum()) for nm in names}
    k = max(1, cfg.state_downsample)
    states: dict[str, np.ndarray] = {}
    for nm in names:
        x0, y0, x1, y1 = bboxes[nm]
        h4, w4 = max(1, (y1 - y0) // k), max(1, (x1 - x0) // k)
        states[nm] = np.zeros((n, h4, w4), dtype=bool)

    # Two chalk-holds with different jobs:
    #  * fullness_hold (plain threshold) feeds the erase detector - a wipe
    #    leaves smears just under the plain threshold, so fullness drops and
    #    the relative-drop detector fires;
    #  * state_hold (wide asymmetric hysteresis: ON > on_thr, OFF < off_thr)
    #    feeds the snapshot states - it keeps flickering borderline pixels and
    #    slowly-brightening eraser smears out of the change diffs. Its OFF
    #    threshold is too forgiving to see wipes, which is exactly why it
    #    cannot drive erase detection.
    thr_on = cfg.chalk_on_threshold
    thr_off = cfg.chalk_off_threshold
    bright_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    fullness_hold = np.zeros((cache.height, cache.width), dtype=bool)
    state_hold = np.zeros((cache.height, cache.width), dtype=bool)

    # YOLO person mask (sampled): the heuristic diff-mask misses dark clothing
    # against dark boards, which poisons chalk-hold and the occlusion signal.
    # The union of both masks is used: YOLO finds him reliably, the heuristic
    # adds fine motion coverage between YOLO samples.
    seg_sampler = None
    if cfg.analysis_person_mask in ("auto", "yolo"):
        try:
            seg = person.YoloPersonSegmenter(cfg.seg_model)
            stride = max(1, round(cfg.analysis_seg_stride_seconds * cfg.analysis_fps))
            seg_sampler = person.SampledPersonMask(
                cache, seg, stride, dilate_px=cfg.person_dilate_px)
        except Exception as exc:
            if cfg.analysis_person_mask == "yolo":
                raise
            print(f"[boards] YOLO analysis mask unavailable ({exc}); heuristic only.")

    for i in tqdm(range(n), desc="analysing frames", unit="f"):
        gray = cache.get(i)
        lvl = fullness.chalk_levels(gray, cfg.chalk_tophat_kernel)
        pmask = person.person_foreground(gray, grid.nearest(i), dilate_px=cfg.person_dilate_px)
        if seg_sampler is not None:
            pmask = pmask | seg_sampler.mask(i)
        # The projector screen is an occluder too: a LARGE bright region (the
        # morphological open removes thin chalk) descending over a board must
        # freeze its state, not erase it - and captures must avoid it.
        screen = cv2.morphologyEx((gray > 170).astype(np.uint8), cv2.MORPH_OPEN,
                                  bright_kernel).astype(bool)
        visible = ~pmask & ~screen
        fullness_hold[visible] = lvl[visible] > cfg.chalk_threshold
        state_hold[(lvl > thr_on) & visible] = True
        state_hold[(lvl < thr_off) & visible] = False
        lap = None
        for nm in names:
            m = masks[nm]
            area = areas[nm]
            if area == 0:
                continue
            occ[nm][i] = 1.0 - int((m & visible).sum()) / area
            person_occ[nm][i] = int((m & pmask).sum()) / area
            raw[nm][i] = np.count_nonzero(fullness_hold & m) / area
            x0, y0, x1, y1 = bboxes[nm]
            states[nm][i] = visits.max_pool((state_hold & m)[y0:y1, x0:x1], k,
                                            min_count=cfg.state_min_count)
            # Sharpness over visible board pixels only - the lecturer's edges
            # otherwise dominate the Laplacian and reward occluded frames.
            if lap is None:
                lap = cv2.Laplacian(gray, cv2.CV_64F)
            region_vis = visible[y0:y1, x0:x1]
            vals = lap[y0:y1, x0:x1][region_vis]
            sharp[nm][i] = float(vals.var()) if vals.size > 100 else 0.0

    states = {nm: visits.debounce_states(states[nm], cfg.state_debounce_frames)
              for nm in names}
    smooth = {nm: fullness.smooth_signal(raw[nm], cfg.fullness_smooth_window) for nm in names}
    return smooth, occ, person_occ, sharp, states


def _column_timeline(nm, f, occ_sig, person_occ_sig, sharp_sig, states, cache, cfg, fps) -> list[_Snapshot]:
    """Emit one snapshot each time the lecturer FINISHES a board and walks away.

    Mimics a student photographing a board once the professor is done with it:
    he stands at a column writing (a "visit", detected from the YOLO-backed
    occlusion signal), then leaves. When he leaves, if the board changed since
    the last snapshot, capture the clean frame now that he is out of the way.
    The crop is later restricted to the single board the new chalk sits on, so
    sliding and the upper/lower boundary never matter here. No erase / lifecycle
    bookkeeping - a wipe-and-rewrite simply produces another finished board.

    A "visit" ends only after he is away for `presence_bridge_seconds`, so a
    brief step-back to look or grab chalk does not trigger a mid-writing capture.
    """
    n = len(f)
    # Presence = the LECTURER only (person_occ), never the projector screen, so
    # a screen lowered over the board is not mistaken for him standing there.
    presence_runs = visits.segment_visits(
        person_occ_sig > cfg.presence_threshold, fps,
        cfg.presence_min_seconds, cfg.presence_bridge_seconds,
    )
    scores = keyframe.score_frames(
        f, occ_sig, sharp_sig,
        cfg.weight_fullness, cfg.weight_occlusion, cfg.weight_sharpness,
    )
    post = max(1, round(cfg.capture_post_seconds * fps))

    ref_state = states[0]
    ref_reset_i = 0
    snaps: list[_Snapshot] = []

    def added_mask(i: int) -> np.ndarray:
        """Clustered cells of fresh chalk at frame i vs the last snapshot."""
        add_m, _rem, _valid = visits.compare_states(states[i], ref_state, max_shift=0)
        return visits.cohesive_mask(add_m)

    def emit(ci: int) -> None:
        nonlocal ref_state, ref_reset_i
        add = added_mask(ci)
        # When did this content appear? From the cumulative added-chalk curve.
        curve = np.array([visits.added_removed(states[j], ref_state, 0)[0]
                          for j in range(ref_reset_i, ci + 1)])
        ws, we = visits.writing_window(curve, floor=0.5 * cfg.snapshot_change_min)
        write_start_i, write_end_i = ref_reset_i + ws, ref_reset_i + we
        spans = [(v.start, v.end) for v in presence_runs
                 if v.end >= write_start_i and v.start <= write_end_i]
        snaps.append(_Snapshot(
            column=nm, write_start_i=write_start_i, write_end_i=write_end_i,
            capture_index=ci, capture_time=float(cache.times[ci]),
            occlusion=float(occ_sig[ci]), added_mask=add, visit_spans=spans,
        ))
        ref_state = states[ci]
        ref_reset_i = ci

    def changed(ci: int) -> bool:
        return int(added_mask(ci).sum()) / states[ci].size >= cfg.snapshot_change_min

    # WALK-AWAY trigger: capture in the gap after each visit, once he is gone.
    for idx, v in enumerate(presence_runs):
        lo = v.end + 1
        hi = min(v.end + post + 1, n)
        nxt = presence_runs[idx + 1].start if idx + 1 < len(presence_runs) else n
        hi = min(hi, nxt)  # never run into his next visit
        if hi <= lo:
            continue
        ci = keyframe.select_best_index(scores, lo, hi)  # cleanest (low-occlusion) frame
        if occ_sig[ci] > cfg.capture_max_occlusion:
            continue  # board still blocked (projector screen / he lingers) - not photographable
        if changed(ci):
            emit(ci)
        else:
            # He left without adding content (erased, or just talked): keep the
            # reference honest so a later rewrite still registers as new.
            _a, rem, valid = visits.compare_states(states[ci], ref_state, 0)
            if int(rem.sum()) / max(1, int(valid.sum())) >= 4 * cfg.snapshot_change_min:
                ref_state, ref_reset_i = states[ci], ci

    # End-of-analysis flush: he finished a board but never left before the clip
    # ended (still standing there). Capture the cleanest available frame.
    if changed(n - 1):
        lo = max(ref_reset_i, n - 1 - post)
        emit(keyframe.select_best_index(scores, lo, n))
    return snaps


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

    smooth, occ, person_occ, sharp, states = _signals(cache, cfg, columns, masks, bboxes, grid)

    snaps: list[_Snapshot] = []
    for c in columns:
        nm = c.name
        snaps.extend(_column_timeline(
            nm, smooth[nm], occ[nm], person_occ[nm], sharp[nm], states[nm], cache, cfg, fps))
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

    k = cfg.state_downsample
    cell_min = cfg.snapshot_change_min  # per-board added-cell fraction to emit a board
    clean_cache: dict[int, np.ndarray] = {}
    records: list[dict] = []
    montage_imgs: list[Path] = []
    montage_caps: list[str] = []
    counters: dict[str, int] = {}
    sid = 0

    for s in tqdm(snaps, desc="exporting boards", unit="capture"):
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

        fx0, fy0, fx1, fy1 = full_bboxes[s.column]
        col_img = clean[fy0:fy1, fx0:fx1]
        col_gray = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
        h_col = fy1 - fy0
        ngrid = s.added_mask.shape[0]

        # Split the column at its wooden rail into the boards visible right now,
        # then emit each board that received fresh chalk as its OWN photo.
        rail_local = enhance.find_rail(col_gray)
        if rail_local is None:
            cells = [(0, h_col, "board")]
        else:
            gr = max(1, min(ngrid - 1, round(rail_local / h_col * ngrid)))
            cells = [(0, rail_local, "upper", slice(0, gr)),
                     (rail_local, h_col, "lower", slice(gr, ngrid))]

        t_a = float(cache.times[s.write_start_i])
        t_b = float(cache.times[s.write_end_i])
        for cell in cells:
            if len(cell) == 4:
                top, bot, vertical, grid_rows = cell
                added_here = int(s.added_mask[grid_rows].sum())
            else:
                top, bot, vertical = cell
                added_here = int(s.added_mask.sum())
            if added_here / s.added_mask.size < cell_min:
                continue  # this board got no real new writing - skip it

            crop = col_img[top:bot]
            location = f"{s.column}_{vertical}" if vertical != "board" else s.column
            sid += 1
            counters[location] = counters.get(location, 0) + 1
            col_dir = boards_dir / s.column
            col_dir.mkdir(exist_ok=True)
            base = f"{vertical}_{counters[location]:02d}_t{tstamp}"
            crop_path = col_dir / f"{base}.{cfg.export_format}"
            cv2.imwrite(str(crop_path), crop)

            enh_rel = None
            if cfg.enhance_crops:
                enh = enhance.enhance_board_crop(crop)
                enh_path = col_dir / f"{base}_enh.{cfg.export_format}"
                cv2.imwrite(str(enh_path), enh)
                enh_rel = enh_path.relative_to(out_dir).as_posix()

            records.append({
                "id": sid,
                "column": s.column,
                "vertical": vertical,
                "location": location,
                "writing_interval": [round(t_a, 1), round(t_b, 1)],
                "writing_interval_str": [_clock(t_a), _clock(t_b)],
                "visits": [[round(float(cache.times[a]), 1), round(float(cache.times[b]), 1)]
                           for a, b in s.visit_spans],
                "capture_time": round(s.capture_time, 1),
                "capture_time_str": _clock(s.capture_time),
                "occlusion": round(s.occlusion, 4),
                "image": crop_path.relative_to(out_dir).as_posix(),
                "image_size": [crop.shape[1], crop.shape[0]],
                "image_enhanced": enh_rel,
                "wall_image": wall_path.relative_to(out_dir).as_posix(),
            })
            montage_imgs.append(crop_path)
            montage_caps.append(f"#{sid} {location} {_clock(s.capture_time)} "
                                f"[wrote {_clock(t_a)}-{_clock(t_b)}]")

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
