"""faster-whisper transcription with automatic GPU->CPU fallback.

CTranslate2 runs a model wholly on one device (no CPU/GPU layer split). With a
2 GB GPU, large-v3 in int8 *may* fit; if it OOMs or the CUDA libraries are
missing, we transparently retry on CPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# On Windows without Developer Mode/admin, HuggingFace's cache symlinks raise
# WinError 1314. Prefer copies and silence the warning so model download is robust.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from ..config import Config
from .prompt import build_initial_prompt


@dataclass
class Attempt:
    device: str
    compute_type: str
    beam_size: int


@dataclass
class TranscriptResult:
    language: str
    duration: float
    device_used: str
    compute_type_used: str
    segments: list[dict] = field(default_factory=list)


def _attempts(cfg: Config, device_override: str | None) -> list[Attempt]:
    device = device_override or cfg.asr_device
    gpu = Attempt("cuda", cfg.asr_compute_type_gpu, cfg.asr_beam_size_gpu)
    cpu = Attempt("cpu", cfg.asr_compute_type_cpu, cfg.asr_beam_size_cpu)
    if device == "cuda":
        return [gpu]
    if device == "cpu":
        return [cpu]
    return [gpu, cpu]  # auto


def _run(audio_path: Path, cfg: Config, attempt: Attempt) -> TranscriptResult:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        cfg.asr_model, device=attempt.device, compute_type=attempt.compute_type
    )
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=cfg.asr_language or None,
        beam_size=attempt.beam_size,
        vad_filter=cfg.asr_vad,
        word_timestamps=cfg.asr_word_timestamps,
        condition_on_previous_text=cfg.asr_condition_on_previous_text,
        initial_prompt=build_initial_prompt(cfg.asr_initial_prompt),
    )

    segments: list[dict] = []
    for seg in segments_iter:  # generator -> work happens here (errors surface here)
        words = None
        if seg.words:
            words = [
                {"start": w.start, "end": w.end, "word": w.word, "prob": w.probability}
                for w in seg.words
            ]
        segments.append(
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": words,
            }
        )

    return TranscriptResult(
        language=info.language,
        duration=info.duration,
        device_used=attempt.device,
        compute_type_used=attempt.compute_type,
        segments=segments,
    )


def transcribe(
    audio_path: str | Path,
    cfg: Config,
    device_override: str | None = None,
) -> TranscriptResult:
    """Transcribe an audio file, trying the configured device(s) in order."""
    audio_path = Path(audio_path)
    attempts = _attempts(cfg, device_override)
    last_err: Exception | None = None

    for idx, attempt in enumerate(attempts):
        try:
            print(f"[asr] attempting device={attempt.device} "
                  f"compute={attempt.compute_type} beam={attempt.beam_size}")
            return _run(audio_path, cfg, attempt)
        except Exception as exc:  # CUDA OOM, missing cuDNN/cuBLAS, etc.
            last_err = exc
            is_last = idx == len(attempts) - 1
            print(f"[asr] device={attempt.device} failed: {type(exc).__name__}: {exc}")
            if is_last:
                break
            print("[asr] falling back to next device...")

    raise RuntimeError(f"All transcription attempts failed. Last error: {last_err}")
