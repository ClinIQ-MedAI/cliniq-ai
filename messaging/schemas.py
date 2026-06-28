"""
Message schemas exchanged between the .NET backend and the ClinIQ AI workers.

Kept as plain dataclasses (no pydantic dependency) so the messaging layer stays
import-light. Everything serializes to/from a flat JSON object.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

# Modalities the AI workers understand. These match the chatbot gateway's
# `image_type` values and the per-service queues (cliniq:jobs:<modality>).
KNOWN_MODALITIES = {
    "bone",
    "dental_xray",
    "dental_photo",
    "chest",
    "prescription",
}


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class JobMessage:
    """A unit of work the backend asks a worker to perform."""

    modality: str
    image_base64: Optional[str] = None      # raw image bytes, base64-encoded
    image_url: Optional[str] = None         # OR a URL the worker fetches
    patient_id: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)  # e.g. {"include_gradcam": true}
    reply_to: Optional[str] = None          # override result channel for this job
    job_id: str = field(default_factory=_new_id)
    enqueued_at: Optional[str] = None        # ISO timestamp, stamped by producer

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "JobMessage":
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        return cls(
            modality=data.get("modality", ""),
            image_base64=data.get("image_base64"),
            image_url=data.get("image_url"),
            patient_id=data.get("patient_id"),
            options=data.get("options") or {},
            reply_to=data.get("reply_to"),
            job_id=data.get("job_id") or _new_id(),
            enqueued_at=data.get("enqueued_at"),
        )


@dataclass
class ResultMessage:
    """The outcome of a job, published back for the backend to consume."""

    job_id: str
    modality: str
    status: str                              # "completed" | "failed"
    result: Optional[Dict[str, Any]] = None  # the predict_for_llm payload
    error: Optional[str] = None
    patient_id: Optional[str] = None
    worker: Optional[str] = None
    model_version: Optional[str] = None      # checkpoint fingerprint (traceability)
    image_sha256: Optional[str] = None       # input fingerprint
    duration_ms: Optional[float] = None
    finished_at: Optional[str] = None        # ISO timestamp, stamped by worker

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "ResultMessage":
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        return cls(**{k: data.get(k) for k in cls.__annotations__})
