"""vbe command-line interface."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from .config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Extract blackboard keyframes and transcripts from lecture videos.")


def _out_dir(out: Path, video: Path) -> Path:
    d = out / video.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.command()
def transcribe(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out: Path = typer.Option(Path("output"), "--out", "-o"),
    device: Optional[str] = typer.Option(None, "--device", help="auto | cuda | cpu"),
    model: Optional[str] = typer.Option(None, "--model", help="override asr_model"),
    start: Optional[float] = typer.Option(None, "--start", help="slice start (s)"),
    duration: Optional[float] = typer.Option(None, "--duration", help="slice length (s)"),
):
    """Transcribe a video's audio to srt/vtt/txt/json."""
    from .io import ffmpeg
    from .transcribe import engine, export

    cfg = load_config(config)
    if model:
        cfg.asr_model = model
    out_dir = _out_dir(out, video)

    typer.echo("[transcribe] extracting audio...")
    wav = ffmpeg.extract_audio(video, out_dir / "_audio.wav", start=start, duration=duration)
    typer.echo(f"[transcribe] running faster-whisper ({cfg.asr_model})...")
    result = engine.transcribe(wav, cfg, device_override=device)
    if start:
        # Shift slice-relative timestamps to absolute video time so they align
        # with keyframe times during pairing.
        for seg in result.segments:
            seg["start"] += start
            seg["end"] += start
            for w in seg.get("words") or []:
                w["start"] = (w["start"] or 0) + start
                w["end"] = (w["end"] or 0) + start
    paths = export.write_all(result, out_dir, basename="transcript")
    typer.echo(f"[transcribe] device={result.device_used} segments={len(result.segments)}")
    for kind, p in paths.items():
        typer.echo(f"  {kind}: {p}")


@app.command()
def boards(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out: Path = typer.Option(Path("output"), "--out", "-o"),
    fps: Optional[float] = typer.Option(None, "--fps", help="override analysis_fps"),
    person_removal: Optional[str] = typer.Option(
        None, "--person-removal", help="median | gated_median | none"),
    start: Optional[float] = typer.Option(None, "--start"),
    duration: Optional[float] = typer.Option(None, "--duration"),
    rebuild: bool = typer.Option(False, "--rebuild", help="ignore cached frames"),
):
    """Extract blackboard keyframes (full-wall + crops + montage + manifest)."""
    from .boards import pipeline
    from .manifest import write_manifest

    cfg = load_config(config)
    if fps:
        cfg.analysis_fps = fps
    if person_removal:
        cfg.person_removal = person_removal
    out_dir = _out_dir(out, video)

    result = pipeline.run_boards(video, cfg, out_dir, start=start, duration=duration, rebuild=rebuild)
    write_manifest(out_dir, result.video, result.keyframes, montage=result.montage)
    typer.echo(f"[boards] {len(result.keyframes)} keyframes -> {out_dir}")
    if result.montage:
        typer.echo(f"[boards] montage: {out_dir / 'montage.png'}")


@app.command(name="boards-roi")
def boards_roi(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    config: Path = typer.Option(..., "--config", "-c", help="config file to write ROIs into"),
    out: Path = typer.Option(Path("output"), "--out", "-o"),
    autodetect: bool = typer.Option(True, "--autodetect/--no-autodetect"),
    fps: float = typer.Option(0.2, "--fps", help="sampling fps for the reference plate"),
    start: Optional[float] = typer.Option(None, "--start"),
    duration: Optional[float] = typer.Option(None, "--duration"),
):
    """Seed board-panel ROIs from a reference plate and write them into a config file."""
    import cv2

    from .boards import roi_autodetect
    from .io import ffmpeg
    from .io.frame_cache import FrameCache

    cfg = load_config(config if config.exists() else None)
    out_dir = _out_dir(out, video)
    info = ffmpeg.probe(video)
    cache = FrameCache.load_or_build(
        video, info, fps=fps, analysis_width=cfg.analysis_width,
        cache_root=out_dir / "_cache", start=start, duration=duration,
    )
    plate = roi_autodetect.reference_plate(cache)
    cv2.imwrite(str(out_dir / "reference_plate.png"), plate)

    rois = roi_autodetect.autodetect_rois(plate, info.width, info.height) if autodetect else []
    typer.echo(f"[boards-roi] detected {len(rois)} panels: {[r.name for r in rois]}")

    data = {}
    if config.exists():
        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    data["rois"] = [{"name": r.name, "polygon": [list(p) for p in r.polygon]} for r in rois]
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    typer.echo(f"[boards-roi] wrote ROIs to {config}")
    typer.echo(f"[boards-roi] review {out_dir / 'reference_plate.png'} and adjust polygons if needed.")


@app.command()
def pair(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("output"), "--out", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    manifest: Optional[Path] = typer.Option(None, "--manifest"),
    transcript: Optional[Path] = typer.Option(None, "--transcript"),
):
    """Attach the transcript text spoken around each keyframe to the manifest."""
    from .pairing import pair_manifest

    cfg = load_config(config)
    out_dir = _out_dir(out, video)
    manifest = manifest or out_dir / "manifest.json"
    transcript = transcript or out_dir / "transcript.json"
    if not manifest.exists():
        raise typer.BadParameter(f"manifest not found: {manifest} (run `vbe boards` first)")
    if not transcript.exists():
        raise typer.BadParameter(f"transcript not found: {transcript} (run `vbe transcribe` first)")
    p = pair_manifest(manifest, transcript, cfg.pair_margin_seconds)
    typer.echo(f"[pair] updated {p}")


@app.command(name="all")
def run_all(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out: Path = typer.Option(Path("output"), "--out", "-o"),
    device: Optional[str] = typer.Option(None, "--device"),
    start: Optional[float] = typer.Option(None, "--start"),
    duration: Optional[float] = typer.Option(None, "--duration"),
):
    """Run boards + transcribe + pair end to end."""
    boards(video, config, out, None, start, duration, False)
    transcribe(video, config, out, device, None, start, duration)
    pair(video, out, config, None, None)


if __name__ == "__main__":
    app()
