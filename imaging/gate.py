"""
Input quality / out-of-distribution (OOD) gate.

A lightweight, **training-free** screen that rejects obviously-wrong uploads
*before* they reach a model — a selfie sent to the bone-fracture detector, a
blank frame, a tiny thumbnail, a screenshot. Without it, a model will happily
return a confident-looking "no findings" on a photo of a cat, which is exactly
the kind of silent failure that embarrasses a medical demo.

It is deliberately **conservative**: it only *rejects* inputs it is confident
are wrong, *flags* borderline ones for human review (the model still runs), and
otherwise gets out of the way — so a genuine scan is never blocked.

Two cheap signals, combined per modality:

* **grayscale-ness** — X-ray modalities (bone/chest/dental X-ray) are near
  grayscale; a saturated colour photograph almost certainly isn't an X-ray.
* **degeneracy** — blank / near-constant / extreme-aspect-ratio / too-small
  images can't be real scans, whatever the modality.

This is a heuristic screen, not a trained OOD detector. A stronger upgrade
(feature-space Mahalanobis / energy score against the model's in-distribution
embeddings) is documented in ``docs/`` as future work; this covers the obvious,
high-value cases with zero model overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# X-ray-like modalities are expected to be (near) grayscale.
GRAYSCALE_MODALITIES = {"bone", "chest", "dental_xray"}
# Ordinary colour photographs / documents — the grayscale test does not apply.
COLOR_MODALITIES = {"dental_photo", "prescription"}

# Thresholds on a 0-255 scale, tuned conservatively to avoid false rejects.
_MIN_SIDE = 100              # smaller than this on a side => not a real scan
_MAX_ASPECT = 5.0            # banner/screenshot-like aspect ratio
_BLANK_STD = 3.0             # intensity std below this => effectively blank

# Grayscale test uses the *fraction of genuinely coloured pixels* rather than a
# mean — a mean is diluted by the neutral regions (walls, shadows) in a real
# photo and by JPEG chroma smoothing, whereas the fraction stays discriminative.
_COLOR_PIXEL_SPREAD = 20     # a pixel whose channel range exceeds this is "coloured"
_COLOR_FRACTION_REJECT = 0.15  # >15% coloured pixels on an X-ray => reject
_COLOR_FRACTION_REVIEW = 0.05  # >5% coloured pixels => run, but flag for review


@dataclass
class GateResult:
    """Verdict of the input gate for one image."""

    passed: bool                 # False => skip the model entirely
    action: str                  # "accept" | "review" | "reject"
    reason: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "action": self.action,
            "reason": self.reason,
            "scores": self.scores,
        }


def check_input(image, modality: str) -> GateResult:
    """
    Screen a PIL image for a given modality.

    Returns a :class:`GateResult`. ``passed=False`` means the caller should skip
    inference and return a rejection; ``action=="review"`` means run the model
    but mark the case for a human to double-check.
    """
    import numpy as np

    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return GateResult(True, "accept", None, {})  # unexpected shape — don't block

    h, w = int(arr.shape[0]), int(arr.shape[1])
    gray = arr.mean(axis=2)
    intensity_std = float(gray.std())
    per_pixel_spread = arr.max(axis=2) - arr.min(axis=2)   # 0 where R==G==B
    color_spread = float(per_pixel_spread.mean())
    colorful_fraction = float((per_pixel_spread > _COLOR_PIXEL_SPREAD).mean())
    aspect = (max(w, h) / min(w, h)) if min(w, h) else 999.0

    scores = {
        "width": w,
        "height": h,
        "aspect_ratio": round(aspect, 2),
        "intensity_std": round(intensity_std, 2),
        "color_spread": round(color_spread, 2),
        "colorful_fraction": round(colorful_fraction, 3),
    }

    # --- Degenerate inputs — rejected for every modality ------------------- #
    if w < _MIN_SIDE or h < _MIN_SIDE:
        return GateResult(False, "reject",
                          f"image too small ({w}x{h}px) to be a real scan", scores)
    if intensity_std < _BLANK_STD:
        return GateResult(False, "reject",
                          "image is blank or near-constant", scores)
    if aspect > _MAX_ASPECT:
        return GateResult(False, "reject",
                          f"unusual aspect ratio ({aspect:.1f}:1); looks like a "
                          f"screenshot or banner, not a scan", scores)

    # --- Grayscale expectation — X-ray modalities only --------------------- #
    if modality in GRAYSCALE_MODALITIES:
        pct = colorful_fraction * 100
        if colorful_fraction > _COLOR_FRACTION_REJECT:
            return GateResult(False, "reject",
                              f"image is in colour ({pct:.0f}% coloured pixels); "
                              f"a {modality.replace('_', ' ')} X-ray should be grayscale",
                              scores)
        if colorful_fraction > _COLOR_FRACTION_REVIEW:
            return GateResult(True, "review",
                              f"unexpected colour for an X-ray ({pct:.0f}% coloured pixels) "
                              f"— flagged for human review", scores)

    return GateResult(True, "accept", None, scores)
