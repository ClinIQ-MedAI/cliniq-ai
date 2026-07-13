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


def _dual_case(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return the dict with both snake_case and camelCase keys.

    Our schemas are snake_case; the .NET backend's model is camelCase
    (chatId, queryType, showUpload, ...). Emitting both means the reply
    deserialises whichever convention the backend's consumer expects, without
    forcing either side to change. Extra keys a parser doesn't know are ignored.
    """
    out = dict(d)
    for k, v in d.items():
        parts = k.split("_")
        camel = parts[0] + "".join(p.title() for p in parts[1:])
        if camel != k and camel not in out:
            out[camel] = v
    return out


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
        return json.dumps(_dual_case(asdict(self)), ensure_ascii=False)

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
        return json.dumps(_dual_case(asdict(self)), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "ResultMessage":
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        return cls(**{k: data.get(k) for k in cls.__annotations__})


@dataclass
class ChatRequest:
    """A single chat turn the backend asks the chatbot to answer.

    Mirrors the chatbot's POST /api/chat body, plus a chat_id the backend uses
    to line the reply up with the conversation it belongs to.
    """

    message: str
    patient_id: str = "anonymous"
    language_preference: str = "ar"          # "ar" | "en"
    chat_id: str = field(default_factory=_new_id)
    reply_to: Optional[str] = None           # override result channel
    enqueued_at: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(_dual_case(asdict(self)), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "ChatRequest":
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        return cls(
            message=data.get("message", ""),
            patient_id=data.get("patient_id") or "anonymous",
            language_preference=data.get("language_preference") or "ar",
            chat_id=data.get("chat_id") or _new_id(),
            reply_to=data.get("reply_to"),
            enqueued_at=data.get("enqueued_at"),
        )


@dataclass
class ChatReply:
    """The chatbot's answer, published back for the backend to consume."""

    chat_id: str
    status: str                              # "completed" | "failed"
    reply: Optional[str] = None              # the full assistant message
    query_type: Optional[str] = None         # health | appointment | faq | ...
    show_upload: bool = False                # the user asked to upload a scan
    patient_id: Optional[str] = None
    error: Optional[str] = None
    worker: Optional[str] = None
    duration_ms: Optional[float] = None
    finished_at: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(_dual_case(asdict(self)), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "ChatReply":
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        return cls(**{k: data.get(k) for k in cls.__annotations__})
