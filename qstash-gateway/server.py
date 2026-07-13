"""QStash gateway — the public HTTP face of the ClinIQ AI stack.

Upstash QStash is push-based: the .NET backend hands a job to QStash, QStash
POSTs it to us over HTTP, we run the prediction synchronously and return the
result in the same HTTP response, and QStash relays that back to the backend.

The internal model services speak multipart file uploads and return raw
predictions; QStash speaks JSON with base64 images and expects a wrapped
envelope. This gateway is the translator, so the model services stay unchanged.

Routes (exactly the two the backend guide specifies):
  POST /predict_for_llm   scans + prescription   (routes by the `modality` field)
  POST /chat              chatbot

Run:  python qstash-gateway/server.py       (defaults to :8080)

Point every AIServiceSettings URL in the backend at this gateway's public URL;
the `modality` field disambiguates, so one gateway serves all five modalities.
"""

from __future__ import annotations

import base64
import binascii
import os
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# modality -> (internal base URL, extra query string)
INTERNAL: Dict[str, str] = {
    "bone":         "http://127.0.0.1:8001",
    "dental_xray":  "http://127.0.0.1:8002",
    "chest":        "http://127.0.0.1:8003",
    "dental_photo": "http://127.0.0.1:8004",
    "prescription": "http://127.0.0.1:8005",
}
CHATBOT_URL = os.getenv("CHATBOT_URL", "http://127.0.0.1:5000")

# The prescription VLM can take minutes on a cold start; everything else is fast.
TIMEOUTS = {"prescription": 1800.0}
DEFAULT_TIMEOUT = 180.0

WORKER = f"{socket.gethostname()}:qstash-gateway"
app = FastAPI(title="ClinIQ QStash Gateway")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----- request bodies QStash sends -------------------------------------------
class ScanJob(BaseModel):
    job_id: str
    modality: str
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    patient_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    reply_to: Optional[str] = None
    enqueued_at: Optional[str] = None


class ChatJob(BaseModel):
    chat_id: str
    message: str
    patient_id: Optional[str] = "anonymous"
    language_preference: Optional[str] = "ar"
    enqueued_at: Optional[str] = None


# ----- helpers ---------------------------------------------------------------
async def _load_image(job: ScanJob) -> bytes:
    """Return the raw image bytes from base64 or a URL."""
    if job.image_base64:
        data = job.image_base64
        if "," in data[:64] and data[:5].lower() == "data:":  # strip data: URI
            data = data.split(",", 1)[1]
        try:
            return base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image_base64 is not valid base64: {exc}") from exc
    if job.image_url:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(job.image_url)
            r.raise_for_status()
            return r.content
    raise ValueError("neither image_base64 nor image_url was provided")


@app.get("/health")
async def health():
    return {"status": "ok", "worker": WORKER, "modalities": list(INTERNAL)}


@app.post("/predict_for_llm")
async def predict_for_llm(job: ScanJob) -> Dict[str, Any]:
    """QStash job -> internal model service -> wrapped envelope."""
    started = time.monotonic()
    base = {
        "job_id": job.job_id,
        "modality": job.modality,
        "patient_id": job.patient_id,
        "worker": WORKER,
    }

    if job.modality not in INTERNAL:
        return {**base, "status": "failed",
                "error": f"unknown modality '{job.modality}'",
                "result": None,
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
                "finished_at": _now()}

    try:
        image_bytes = await _load_image(job)

        # chest reads include_gradcam as a query param; default on.
        params = {}
        if job.modality == "chest":
            want = (job.options or {}).get("include_gradcam", True)
            params["include_gradcam"] = "true" if want else "false"

        url = f"{INTERNAL[job.modality]}/predict_for_llm"
        timeout = TIMEOUTS.get(job.modality, DEFAULT_TIMEOUT)
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}

        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, files=files, params=params)
            r.raise_for_status()
            result = r.json()

        return {**base, "status": "completed", "result": result, "error": None,
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
                "finished_at": _now()}

    except Exception as exc:  # noqa: BLE001 — any failure becomes a failed envelope
        return {**base, "status": "failed", "result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
                "finished_at": _now()}


@app.post("/chat")
async def chat(job: ChatJob) -> Dict[str, Any]:
    """QStash chat turn -> chatbot /api/chat (NDJSON stream folded to one reply)."""
    started = time.monotonic()
    base = {"chat_id": job.chat_id, "patient_id": job.patient_id, "worker": WORKER}

    try:
        import json

        text_parts: list[str] = []
        meta: Dict[str, Any] = {}
        payload = {
            "message": job.message,
            "patient_id": job.patient_id,
            "language_preference": job.language_preference,
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            async with client.stream("POST", f"{CHATBOT_URL}/api/chat",
                                     json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if "chunk" in obj:
                        text_parts.append(obj["chunk"])
                    if obj.get("done"):
                        meta = obj

        return {**base, "status": "completed",
                "reply": "".join(text_parts),
                "query_type": meta.get("query_type"),
                "show_upload": bool(meta.get("show_upload")),
                "error": None,
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
                "finished_at": _now()}

    except Exception as exc:  # noqa: BLE001
        return {**base, "status": "failed", "reply": None,
                "query_type": None, "show_upload": False,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
                "finished_at": _now()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("QSTASH_GATEWAY_PORT", "8080")))
