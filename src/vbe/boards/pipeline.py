"""Blackboard keyframe extraction orchestrator.

End to end: decode -> analysis-resolution fullness/occlusion/sharpness signals ->
per-panel erase/slide epoch detection -> pick the fullest, least-occluded moments
-> export full-resolution lecturer-removed keyframes + crops + montage + records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..config import Config
from ..io import ffmpeg
from ..io.frame_cache import FrameCache
from . import background, dedup, epochs, fullness, keyframe, montage, person, roi


def _fmt_t(seconds: float) -> str:
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}-{m:02d}-{s:02d}"


@dataclass
class BoardsResult:
    video: str
    keyframes: list[dict] = field(default_factory=list)
    montage: str | None = None


def _global_color_plate(video_path, times: np.ndarray, k: int = 41) -> np.ndarray:
    """Full-resolution, person-free color plate: median over frames sampled across
    the whole analysis range in a single low-fps ffmpeg pass."""
    t0, t1 = float(times[0]), float(times[-1])
    span = max(t1 - t0, 1.0)
    fps_g = min(1.0, k / span)
    frames = [f for _, f in ffmpeg.decode_frames(
        video_path, fps=fps_g, start=t0, duration=span, size=None, gray=False)]
    if not frames:
        return None  # type: ignore[return-value]
    return background.median_plate(np.stack(frames, axis=0))


def _cluster(indices: list[int], gap: int) -> list[list[int]]:
    if not indices:
        return []
    indices = sorted(set(indices))
    groups = [[indices[0]]]
    for i in indices[1:]:
        if i - groups[-1][-1] <= gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def run_boards(
    video_path: str | Path,
    cfg: Config,
    out_dir: str | Path,
    start: float | None = None,
    duration: float | None = None,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
) -> BoardsResult:
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
    scale = cache.width / info.width
    n = cache.n
    if n == 0:
        return BoardsResult(video=video_path.name)

    # --- ROIs (panels + synthetic full-wall) ---
    panels = roi.panel_rois(cfg.rois)
    full = roi.default_full_roi(info.width, info.height)
    rois_all = panels + [full]
    masks = {r.name: roi.roi_mask(r, cache.height, cache.width, scale) for r in rois_all}
    bboxes = {r.name: roi.scaled_bbox(r, scale, cache.width, cache.height) for r in rois_all}

    # --- occlusion reference plates ---
    half = max(1, round(cfg.median_window_seconds * fps))
    grid = background.build_plate_grid(cache, half_window_frames=half)

    # --- single analysis pass: fullness / occlusion / sharpness ---
    names = [r.name for r in rois_all]
    full_signal = {nm: np.zeros(n) for nm in names}
    occ_signal = {nm: np.zeros(n) for nm in names}
    sharp = np.zeros(n)

    for i in tqdm(range(n), desc="analysing frames", unit="f"):
        gray = cache.get(i)
        chalk = fullness.chalk_mask(gray, cfg.chalk_tophat_kernel, cfg.chalk_threshold)
        pmask = person.person_foreground(gray, grid.nearest(i), dilate_px=cfg.person_dilate_px)
        sharp[i] = cv2.Laplacian(gray, cv2.CV_64F).var()
        for nm in names:
            m = masks[nm]
            area = int(m.sum())
            full_signal[nm][i] = (np.count_nonzero(chalk & m) / area) if area else 0.0
            occ_signal[nm][i] = (np.count_nonzero(pmask & m) / area) if area else 0.0

    smooth = {nm: fullness.smooth_signal(full_signal[nm], cfg.fullness_smooth_window) for nm in names}

    presnap = max(1, round(cfg.keyframe_presnap_seconds * fps))
    refractory = max(1, round(cfg.erase_refractory_seconds * fps))
    slide_px = cfg.slide_shift_px * scale

    # --- per-panel epoch detection -> candidate capture indices ---
    detection_rois = panels if panels else [full]
    candidates: list[int] = []
    panel_events: dict[str, list[epochs.Event]] = {}
    for r in detection_rois:
        nm = r.name
        evs = epochs.detect_events(smooth[nm], cfg.erase_min_drop, refractory)
        # classify slide vs erase using image data
        x0, y0, x1, y1 = bboxes[nm]
        for ev in evs:
            nxt = min(ev.drop_index + presnap, n - 1)
            prev_roi = cache.get(ev.peak_index)[y0:y1, x0:x1]
            next_roi = cache.get(nxt)[y0:y1, x0:x1]
            is_slide, _dy = epochs.classify_slide(prev_roi, next_roi, slide_px)
            ev.kind = "slide" if is_slide else "erase"
        panel_events[nm] = evs

        cap = epochs.capture_indices(smooth[nm], evs)
        scores = keyframe.score_frames(
            full_signal[nm], occ_signal[nm], sharp,
            cfg.weight_fullness, cfg.weight_occlusion, cfg.weight_sharpness,
        )
        for ci in cap:
            best = keyframe.select_best_index(scores, ci - presnap, ci + presnap + 1)
            candidates.append(best)

    # --- cluster candidate moments into full-wall keyframes ---
    groups = _cluster(candidates, gap=presnap)
    occ_full = occ_signal["full"]
    reps = [min(g, key=lambda i: occ_full[i]) for g in groups]

    # --- export clean full-res keyframes ---
    segmenter = None
    if cfg.person_removal == "gated_median":
        try:
            segmenter = person.YoloPersonSegmenter(cfg.seg_model)
        except Exception as exc:  # [seg] not installed / model missing
            print(f"[boards] gated_median requested but segmenter unavailable ({exc}); "
                  f"using median compositing.")

    import shutil
    keyframes_dir = out_dir / "keyframes"
    if keyframes_dir.exists():
        shutil.rmtree(keyframes_dir)  # avoid mixing stale keyframes from prior runs
    full_dir = keyframes_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    ext = cfg.export_format

    # Person-free full-resolution global plate (whole-range median) for infill.
    global_plate = _global_color_plate(video_path, cache.times)

    cand_objs: list[dedup.KeyframeCandidate] = []
    for r in tqdm(reps, desc="exporting keyframes", unit="kf"):
        t = float(cache.times[r])
        clean = keyframe.build_clean_frame(
            video_path, t, cfg.median_window_seconds, fps,
            global_plate=global_plate,
            dilate_px=cfg.person_dilate_px + 2, segmenter=segmenter,
        )
        cand_objs.append(dedup.KeyframeCandidate(
            index=r, time=t, image=clean,
            fullness=float(full_signal["full"][r]),
            occlusion=float(occ_full[r]),
            meta={"panels_present": [p.name for p in panels]},
        ))

    kept = dedup.deduplicate(cand_objs, cfg.phash_max_distance, cfg.ssim_tiebreak)

    # --- write images, crops, montage, records ---
    records: list[dict] = []
    montage_paths: list[Path] = []
    montage_times: list[float] = []
    for k, cand in enumerate(kept, start=1):
        base = f"epoch_{k:04d}_t{_fmt_t(cand.time)}"
        full_path = full_dir / f"{base}.{ext}"
        cv2.imwrite(str(full_path), cand.image)
        montage_paths.append(full_path)
        montage_times.append(cand.time)

        crop_paths: list[str] = []
        if cfg.emit_region_crops and panels:
            for p in panels:
                x0, y0, x1, y1 = roi.fullres_bbox(p, info.width, info.height)
                crop = cand.image[y0:y1, x0:x1]
                cdir = out_dir / "keyframes" / "crops" / p.name
                cdir.mkdir(parents=True, exist_ok=True)
                cpath = cdir / f"{base}.{ext}"
                cv2.imwrite(str(cpath), crop)
                crop_paths.append(str(cpath.relative_to(out_dir)))

        records.append({
            "id": k,
            "time": cand.time,
            "time_str": _fmt_t(cand.time).replace("-", ":"),
            "image": str(full_path.relative_to(out_dir)),
            "crops": crop_paths,
            "fullness": round(cand.fullness, 4),
            "occlusion": round(cand.occlusion, 4),
        })

    montage_path = None
    if montage_paths:
        mp = montage.make_montage(montage_paths, montage_times, out_dir / "montage.png")
        montage_path = str(mp.relative_to(out_dir))

    return BoardsResult(video=video_path.name, keyframes=records, montage=montage_path)
