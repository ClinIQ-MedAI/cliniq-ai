"""
Model version fingerprinting.

A model "version" is a stable, reproducible id derived from the weights file
itself — so every logged prediction can be traced back to the exact checkpoint
that produced it, even if filenames are reused. Format:

    <stem>@<sha12> (<size_mb>MB)
    e.g.  best.pt@a1b2c3d4e5f6 (114.4MB)

The sha is over the file's size + first/last 1 MiB (not the whole multi-GB file),
which is fast and still uniquely identifies a checkpoint in practice.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional

# Root of the consolidated models tree (cliniq/models/...).
_CLINIQ_ROOT = Path(__file__).resolve().parents[1]
_MODELS = _CLINIQ_ROOT / "models"

# modality -> weights file used by that service's primary model.
KNOWN_MODEL_PATHS: Dict[str, Path] = {
    "bone": _MODELS / "bone-detect" / "YOLO11x_TOP3_20260203_0645" / "weights" / "best.pt",
    "dental_xray": _MODELS / "oral-xray" / "yolo_v8x_base_1024" / "weights" / "best.pt",
    "dental_photo": _MODELS / "oral-classify" / "SOTA_FINAL_20251124_1300" / "best_model.pth",
    "chest": _CLINIQ_ROOT / "chest_xray" / "outputs" / "checkpoints" / "best.pt",
    # prescription-parser pulls its VLM from HuggingFace at runtime; no local file.
    "prescription": None,
}

_CACHE: Dict[str, str] = {}


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(chunk))           # first 1 MiB
        if size > chunk:
            fh.seek(max(0, size - chunk))
            h.update(fh.read(chunk))       # last 1 MiB
    return h.hexdigest()


def model_version(path: Optional[os.PathLike | str]) -> str:
    """Return a stable version string for a weights file (or 'unknown')."""
    if not path:
        return "unknown"
    p = Path(path)
    key = str(p)
    if key in _CACHE:
        return _CACHE[key]
    try:
        if not p.exists():
            version = f"{p.name}@missing"
        else:
            sha = _hash_file(p)[:12]
            size_mb = p.stat().st_size / (1024 * 1024)
            version = f"{p.name}@{sha} ({size_mb:.1f}MB)"
    except Exception as exc:  # noqa: BLE001
        version = f"{p.name}@error:{exc}"
    _CACHE[key] = version
    return version


def model_version_for_modality(modality: str) -> str:
    """Fingerprint the primary model for a modality, or a runtime label."""
    if modality == "prescription":
        return "Qwen2-VL-72B-AWQ@huggingface-runtime"
    return model_version(KNOWN_MODEL_PATHS.get(modality))
