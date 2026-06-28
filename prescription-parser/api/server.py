"""
Prescription Parser API server (port 8005).

Exposes the PrescriptionParserService over HTTP so the chatbot can route
handwritten prescription images to it.
"""

from __future__ import annotations

import os

# Avoid pulling TensorFlow (which has a stale protobuf) through transformers,
# and use the pure-Python protobuf parser as a safety net.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Allow `from prescription_pipeline import ...` whether launched as
# `python api/server.py` or as a package.
SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from prescription_pipeline import PrescriptionParserService  # noqa: E402

logger = logging.getLogger("prescription_parser.api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(title="ClinIQ Prescription Parser", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Lightweight readiness probe — does not load the heavy VLM."""
    return {"status": "ok", "service": "prescription-parser", "port": 8005}


@app.get("/status")
def status() -> Dict[str, Any]:
    """Live status of the prescription parser (loading stages, inference, etc.)."""
    service = PrescriptionParserService.get_instance()
    return service.get_status()


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        data = upload.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload.")
        return Image.open(io.BytesIO(data)).convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


@app.post("/parse")
async def parse(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Parse a prescription image and return the verified medications JSON."""
    image = _read_image(file)
    service = PrescriptionParserService.get_instance()
    try:
        result = service.parse(image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.to_dict()


@app.post("/predict_for_llm")
async def predict_for_llm(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Chatbot-friendly response shape (mirrors other ClinIQ services).

    Returns:
        {
            "success": bool,
            "image_type": "prescription",
            "detections": [],            # no bounding boxes for prescriptions
            "ai_findings": {
                "primary_diagnosis": "...",
                "medications": [ {drug, dosage, frequency, official_match, confidence_score}, ... ]
            },
            "report_data": {...}
        }
    """
    image = _read_image(file)
    service = PrescriptionParserService.get_instance()
    try:
        result = service.parse(image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed")
        return {
            "success": False,
            "image_type": "prescription",
            "error": str(exc),
            "detections": [],
            "ai_findings": {},
        }

    meds = result.medications
    verified = [m for m in meds if m.get("official_match")]
    primary = (
        f"تم استخراج {len(meds)} دواء من الروشتة "
        f"({len(verified)} منها متحقق رسمياً ضمن قاعدة الأدوية المصرية)."
        if meds
        else "لم يتم العثور على أدوية واضحة في الصورة."
    )

    return {
        "success": True,
        "image_type": "prescription",
        "detections": [],
        "ai_findings": {
            "primary_diagnosis": primary,
            "medications": meds,
            "raw_vlm_output": result.raw_vlm_output,
            "notes": result.notes,
        },
        "report_data": {
            "total_medications": len(meds),
            "verified_medications": len(verified),
            "medications": meds,
        },
    }


# ==================== ASYNC QUEUE WORKER (opt-in) ====================
# Consumes `cliniq:jobs:prescription` and publishes to `cliniq:results` when
# QUEUE_BACKEND is set. Reuses predict_for_llm. No-op otherwise.
import sys as _sys

_CLINIQ_ROOT = Path(__file__).resolve().parents[2]
if str(_CLINIQ_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CLINIQ_ROOT))
try:
    from messaging.fastapi_integration import attach_worker

    attach_worker(app, modality="prescription", route=predict_for_llm)
except Exception as _queue_exc:  # noqa: BLE001
    print(f"[prescription-parser] queue worker not attached: {_queue_exc}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info")
