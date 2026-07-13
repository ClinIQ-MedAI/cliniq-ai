"""
DICOM decoding for the ClinIQ imaging pipeline.

Radiology scans usually leave a PACS as DICOM (``.dcm``), not PNG/JPEG. This
module detects DICOM bytes and transcodes them into a display-ready 8-bit RGB
image that the existing model preprocessing (``PIL.Image.open``) consumes
unchanged. It applies the standard radiology display pipeline:

    Modality LUT (rescale slope/intercept)  ->  real units (e.g. HU)
    VOI LUT / windowing (WindowCenter/Width) ->  display range
    MONOCHROME1 inversion                     ->  bone/tissue reads normally
    min-max normalization                     ->  0..255 uint8
    grayscale -> RGB                          ->  3-channel for the models

``pydicom`` is treated as an optional dependency: :func:`is_dicom` works by
magic-byte sniffing even without it, and :func:`decode_dicom` raises a clear,
actionable :class:`DicomError` if the library (or a codec plugin) is missing —
so a service that never receives a DICOM keeps running even if pydicom isn't
installed.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Tuple

# A conformant DICOM file carries the ASCII magic "DICM" at byte offset 128,
# right after a 128-byte preamble. That lets us detect one without a full parse.
_DICM_MAGIC = b"DICM"
_DICM_OFFSET = 128


class DicomError(RuntimeError):
    """Raised when DICOM bytes cannot be decoded (missing lib, bad codec, ...)."""


def is_dicom(data: bytes) -> bool:
    """
    True if ``data`` looks like a DICOM file.

    Strict check on the ``DICM`` magic at offset 128 — regular PNG/JPEG/WebP
    never match there, so this does not false-positive on ordinary images.
    Preamble-less DICOM streams are handled at the call site via the filename
    hint (see :func:`imaging.normalize_medical_image`).
    """
    return (
        bool(data)
        and len(data) >= _DICM_OFFSET + 4
        and data[_DICM_OFFSET:_DICM_OFFSET + 4] == _DICM_MAGIC
    )


def _extract_metadata(ds) -> Dict[str, Any]:
    """Pull the clinically useful, non-pixel header fields for audit/traceability."""

    def _s(tag: str):
        val = getattr(ds, tag, None)
        return str(val).strip() if val not in (None, "") else None

    def _i(tag: str):
        val = getattr(ds, tag, None)
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    transfer_syntax = None
    file_meta = getattr(ds, "file_meta", None)
    if file_meta is not None:
        ts = getattr(file_meta, "TransferSyntaxUID", None)
        transfer_syntax = str(ts) if ts else None

    return {
        "is_dicom": True,
        "modality": _s("Modality"),                    # CR, DX, CT, MR, US, ...
        "body_part": _s("BodyPartExamined"),
        "study_date": _s("StudyDate"),
        "patient_id": _s("PatientID"),
        "patient_sex": _s("PatientSex"),
        "patient_age": _s("PatientAge"),
        "rows": _i("Rows"),
        "columns": _i("Columns"),
        "photometric": _s("PhotometricInterpretation"),
        "manufacturer": _s("Manufacturer"),
        "transfer_syntax": transfer_syntax,
    }
    # NOTE: PatientName is intentionally omitted — it is direct PHI and the
    # platform already keys everything on the pseudonymous patient_id.


def _load_luts():
    """Import apply_modality_lut / apply_voi_lut across pydicom 2.x and 3.x."""
    try:  # pydicom >= 3.0
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
        return apply_modality_lut, apply_voi_lut
    except Exception:  # pragma: no cover - older pydicom
        from pydicom.pixel_data_handlers.util import (
            apply_modality_lut,
            apply_voi_lut,
        )
        return apply_modality_lut, apply_voi_lut


def decode_dicom(data: bytes) -> Tuple["Image.Image", Dict[str, Any]]:  # noqa: F821
    """
    Decode DICOM bytes into ``(PIL.Image RGB, metadata)``.

    Raises :class:`DicomError` if pydicom is not installed or the pixel data
    uses a compressed transfer syntax whose codec plugin is missing.
    """
    try:
        import numpy as np
        import pydicom
    except ImportError as exc:
        raise DicomError(
            "DICOM support requires pydicom. Install it with:\n"
            "    pip install pydicom pylibjpeg pylibjpeg-libjpeg"
        ) from exc
    from PIL import Image

    try:
        ds = pydicom.dcmread(io.BytesIO(data), force=True)
    except Exception as exc:  # noqa: BLE001
        raise DicomError(f"Not a readable DICOM file: {exc}") from exc

    meta = _extract_metadata(ds)

    try:
        arr = ds.pixel_array
    except Exception as exc:  # noqa: BLE001 - usually a missing codec plugin
        raise DicomError(
            f"Could not decode DICOM pixel data "
            f"(transfer syntax: {meta.get('transfer_syntax') or 'unknown'}). "
            "For compressed DICOM install a codec plugin:\n"
            "    pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg"
        ) from exc

    apply_modality_lut, apply_voi_lut = _load_luts()

    # Stored values -> real modality units (HU for CT, etc.). No-op if absent.
    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:  # noqa: BLE001
        pass

    # Windowing to a display range using WindowCenter/WindowWidth or a VOI LUT.
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:  # noqa: BLE001
        pass

    arr = np.asarray(arr, dtype=np.float32)

    # Multi-frame series (e.g. CT/MR volumes): take the middle slice for a 2D
    # preview. Grayscale multi-frame is (frames, rows, cols); color is (..,3/4).
    if arr.ndim == 4:
        arr = arr[arr.shape[0] // 2]
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[arr.shape[0] // 2]

    is_grayscale = arr.ndim == 2

    # MONOCHROME1: minimum value is white -> invert so it reads like a film.
    if is_grayscale and str(meta.get("photometric") or "").upper() == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB"), meta


def dicom_to_png_bytes(data: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Decode DICOM and re-encode as PNG bytes; returns ``(png_bytes, metadata)``."""
    image, meta = decode_dicom(data)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue(), meta
