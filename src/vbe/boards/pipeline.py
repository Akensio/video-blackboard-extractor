"""Board-timeline extraction orchestrator.

Per column of boards (the camera wall has three), this builds the lecture's
writing timeline:

  decode low-fps frames -> per-column chalk-hold state + occlusion signals
  -> lecturer VISITS (sustained presence at a column) and erase events
  -> snapshot triggers: visit ends, imminent erases, quiet-time flushes -
     each gated by "did the column's chalk actually change since the last
     snapshot" (slide-compensated diff, so moved-but-unchanged content
     doesn't count)
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
from . import background, enhance, epochs, fullness, keyframe, montage, person, roi, visits


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
    write_start_i: int        # analysis-frame index: content began appearing
    write_end_i: int          # analysis-frame index: content was complete
    capture_index: int
    capture_time: float
    fullness: float
    occlusion: float
    trigger: str              # visit_end | pre_erase | flush
    visit_spans: list[tuple[int, int]] = field(default_factory=list)
    erased_at: float | None = None
    final: bool = False


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
    occ = {nm: np.zeros(n) for nm in names}
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
    return smooth, occ, sharp, states


def _column_timeline(nm, f, occ_sig, sharp_sig, states, cache, cfg, fps) -> list[_Snapshot]:
    """Settle / pre-erase / flush triggers -> change-gated snapshots for one column.

    The trigger watches the BOARD, not the lecturer: when the column's chalk
    state has changed since the last snapshot and then stops changing for
    `stable_seconds`, the content is "settled" and gets captured. (His dark
    clothing makes position-based triggering unreliable; presence is recorded
    only as informational `visits`.)
    """
    n = len(f)
    refractory = max(1, round(cfg.erase_refractory_seconds * fps))
    persist = max(1, round(cfg.erase_persist_seconds * fps))
    events = epochs.detect_events(
        f, cfg.erase_min_drop, refractory,
        min_peak=cfg.erase_min_peak, persist_frames=persist,
    )
    erase_idx = [e.drop_index for e in events]

    presence_runs = visits.segment_visits(
        occ_sig > cfg.presence_threshold, fps,
        cfg.presence_min_seconds, cfg.presence_bridge_seconds,
    )

    scores = keyframe.score_frames(
        f, occ_sig, sharp_sig,
        cfg.weight_fullness, cfg.weight_occlusion, cfg.weight_sharpness,
    )
    post = max(1, round(cfg.capture_post_seconds * fps))
    back = max(1, round(cfg.capture_preerase_back_seconds * fps))
    stable_w = max(1, round(cfg.stable_seconds * fps))
    max_shift = max(2, states.shape[1] // 2)  # boards slide up to ~half the column

    ref_state = states[0]
    ref_reset_i = 0
    snaps: list[_Snapshot] = []

    def gate(i: int) -> str:
        """Classify the pending change at frame i.

        'emit'  - the diff contains a substantial CLUSTERED added-chalk set:
                  real writing forms line clusters, so its cohesive count is
                  ~its full size, while drying smears and slide-compensation
                  residue are scattered and contribute almost nothing. Misses
                  are catastrophic and redundant snapshots are cheap, so this
                  is deliberately the only emit condition;
        'reset' - remove-dominant change with no written content pending: a
                  partial erase the column-level detector missed - re-reference
                  silently, there is nothing to save;
        'none'  - nothing significant pending.
        """
        add_m, rem_m, valid = visits.compare_states(states[i], ref_state, max_shift)
        n = max(1, int(valid.sum()))
        add_cohesive = visits.cohesive_count(add_m) / n
        rem = int(rem_m.sum()) / n
        if add_cohesive >= cfg.snapshot_change_min:
            # Known benign artifact: a wipe drying ACROSS the analysis start
            # can emit one empty-board snapshot (its smears brighten into
            # cohesive streaks). Image-level smear gates were tried and
            # rejected - they can also kill real sparse writing, and a
            # redundant frame is cheap while lost content is not.
            return "emit"
        if rem >= 4 * cfg.snapshot_change_min:
            return "reset"
        return "none"

    def emit(ci: int, trigger: str) -> None:
        nonlocal ref_state, ref_reset_i
        board_index = 1 + sum(1 for e in erase_idx if e <= ci)
        # When did this content appear? From the cumulative ADDED-chalk curve.
        curve = np.array([visits.added_removed(states[j], ref_state, max_shift)[0]
                          for j in range(ref_reset_i, ci + 1)])
        ws, we = visits.writing_window(curve, floor=0.5 * cfg.snapshot_change_min)
        write_start_i, write_end_i = ref_reset_i + ws, ref_reset_i + we
        spans = [(v.start, v.end) for v in presence_runs
                 if v.end >= write_start_i and v.start <= write_end_i]
        nxt_drop = next((e for e in erase_idx if e > ci), None)
        snaps.append(_Snapshot(
            column=nm, board_index=board_index,
            write_start_i=write_start_i, write_end_i=write_end_i,
            capture_index=ci, capture_time=float(cache.times[ci]),
            fullness=float(f[ci]), occlusion=float(occ_sig[ci]),
            trigger=trigger,
            visit_spans=spans,
            erased_at=float(cache.times[nxt_drop]) if nxt_drop is not None else None,
        ))
        ref_state = states[ci]
        ref_reset_i = ci

    next_event = 0
    i = stable_w
    while i < n:
        # Process any erase whose drop we just reached: capture the pre-wipe
        # state from a backward window ending at the fullness peak. The gate
        # is evaluated at the PEAK (is there content worth saving), while the
        # photo is the best-scored frame in the backward window.
        if next_event < len(events) and events[next_event].drop_index <= i:
            ev = events[next_event]
            next_event += 1
            lo = max(ref_reset_i + 1, ev.peak_index - back)
            hi = ev.peak_index + 1
            if hi > lo and gate(ev.peak_index) == "emit":
                ci = keyframe.select_best_index(scores, lo, hi)
                emit(ci, "pre_erase")
            # The wipe invalidates the reference whether or not we emitted.
            j = min(ev.drop_index + persist, n - 1)
            ref_state = states[j]
            ref_reset_i = j
            i = max(i, j + 1)
            continue

        settled = (np.count_nonzero(states[i] ^ states[i - stable_w])
                   / states[i].size) < cfg.stable_eps
        if settled:
            # Defer while the column is blocked (projector screen down, or the
            # lecturer camped in front): the pending change persists, and the
            # capture fires at the next unblocked settle point.
            if occ_sig[i] > cfg.capture_max_occlusion:
                i += 1
                continue
            g = gate(i)
            if g == "emit":
                hi = min(i + post + 1, n)
                nxt_ev = next((e for e in events if e.drop_index > i), None)
                if nxt_ev is not None:
                    hi = min(hi, max(nxt_ev.peak_index + 1, i + 1))
                ci = keyframe.select_best_index(scores, i, max(hi, i + 1))
                emit(ci, "settled")
                i = max(i + 1, ci + 1)
                continue
            if g == "reset":
                ref_state = states[i]
                ref_reset_i = i
        i += 1

    # End-of-analysis flush: capture whatever changed but never settled.
    lo = max(ref_reset_i, n - 1 - post)
    ci = keyframe.select_best_index(scores, lo, n)
    if gate(ci) == "emit":
        emit(ci, "flush")

    # final = last snapshot of each board in this column
    for i, s in enumerate(snaps):
        s.final = not any(t.board_index == s.board_index for t in snaps[i + 1:])
    return snaps


def _merge_near_duplicates(snaps: list[_Snapshot], cache: FrameCache, bboxes,
                           phash_max: int) -> list[_Snapshot]:
    """Backstop: within a column, merge consecutive snapshots whose content is
    near-identical. The later snapshot wins; its writing interval extends back."""
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
                out[j] = _Snapshot(
                    column=s.column, board_index=s.board_index,
                    write_start_i=prev.write_start_i, write_end_i=s.write_end_i,
                    capture_index=s.capture_index, capture_time=s.capture_time,
                    fullness=s.fullness, occlusion=s.occlusion,
                    trigger=s.trigger,
                    visit_spans=prev.visit_spans + s.visit_spans,
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

    smooth, occ, sharp, states = _signals(cache, cfg, columns, masks, bboxes, grid)

    snaps: list[_Snapshot] = []
    for c in columns:
        nm = c.name
        snaps.extend(_column_timeline(
            nm, smooth[nm], occ[nm], sharp[nm], states[nm], cache, cfg, fps))
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
        crop = clean[y0:y1, x0:x1]
        if cfg.trim_crops:
            crop = enhance.trim_to_board(crop)
        crop_path = col_dir / f"{base}.{cfg.export_format}"
        cv2.imwrite(str(crop_path), crop)

        enh_rel = None
        if cfg.enhance_crops:
            enh = enhance.enhance_board_crop(crop)
            enh_path = col_dir / f"{base}_enh.{cfg.export_format}"
            cv2.imwrite(str(enh_path), enh)
            enh_rel = enh_path.relative_to(out_dir).as_posix()

        t_a = float(cache.times[s.write_start_i])
        t_b = float(cache.times[s.write_end_i])
        records.append({
            "id": sid,
            "column": s.column,
            "board": f"{s.column}#{s.board_index}",
            "writing_interval": [round(t_a, 1), round(t_b, 1)],
            "writing_interval_str": [_clock(t_a), _clock(t_b)],
            "visits": [[round(float(cache.times[a]), 1), round(float(cache.times[b]), 1)]
                       for a, b in s.visit_spans],
            "trigger": s.trigger,
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
            + (" pre-erase" if s.trigger == "pre_erase" else "")
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
