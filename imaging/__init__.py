"""
ClinIQ shared imaging utilities.

Currently exposes the DICOM front door used by both the async queue workers and
the chatbot gateway. The single entry point is :func:`normalize_medical_image`,
which turns *any* uploaded bytes into something the models already understand
(PNG/JPEG bytes) plus an optional metadata dict — so no downstream service code
needs to know what DICOM is.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .dicom import DicomError, decode_dicom, dicom_to_png_bytes, is_dicom
from .gate import GateResult, check_input

__all__ = [
    "normalize_medical_image",
    "is_dicom",
    "decode_dicom",
    "dicom_to_png_bytes",
    "DicomError",
    "check_input",
    "GateResult",
]

_DICOM_SUFFIXES = (".dcm", ".dicom", ".ima")


def normalize_medical_image(
    data: bytes, filename: Optional[str] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Normalize an uploaded medical image to display-ready bytes.

    * If ``data`` is DICOM (by magic bytes, or by a ``.dcm``/``.dicom`` filename
      hint for preamble-less streams), transcode it to PNG and return the DICOM
      header metadata.
    * Otherwise return the original bytes untouched with an empty metadata dict.

    Returns ``(image_bytes, metadata)``. Downstream code keeps doing
    ``Image.open(BytesIO(image_bytes))`` exactly as before.

    Raises :class:`DicomError` only when the input *is* DICOM but cannot be
    decoded (pydicom missing, or an unsupported compressed transfer syntax);
    callers should surface that as a rejected job rather than crashing.
    """
    name = (filename or "").lower()
    looks_dicom = is_dicom(data) or name.endswith(_DICOM_SUFFIXES)
    if not looks_dicom:
        return data, {}
    return dicom_to_png_bytes(data)
