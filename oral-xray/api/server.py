"""
Dental X-ray API server (two-stage YOLO + ConvNeXt classifier).

Wraps the original DentalInferencePipeline so the chatbot gets back the rich
class set (Wisdom Tooth, Decay, Missing Tooth, Dental Filling, Root Canal
Filling, Implant, Porcelain Crown, Ceramic Bridge, Apical Periodontitis)
instead of the 4-class detector-only output.
"""

import base64
import io
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont

# Make the original two-stage pipeline importable.
# Paths are relative to the consolidated `cliniq/models/oral-xray` tree.
MODELS_ROOT = Path(__file__).resolve().parents[2] / "models" / "oral-xray"
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from src.inference.pipeline import DentalInferencePipeline  # noqa: E402

YOLO_WEIGHTS = str(MODELS_ROOT / "yolo_v8x_base_1024/weights/best.pt")
CLASSIFIER_WEIGHTS = str(MODELS_ROOT / "convnext_large_20260130_090637/weights/best.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

YOLO_CONF = 0.20
CLASSIFIER_REFINE_CONF = 0.70

SEVERITY_MAP = {
    "Decay": "HIGH",
    "Apical Periodontitis": "HIGH",
    "Missing Tooth": "MEDIUM",
    "Wisdom Tooth": "LOW",
    "Dental Filling": "LOW",
    "Root Canal Filling": "LOW",
    "Implant": "LOW",
    "Porcelain Crown": "LOW",
    "Ceramic Bridge": "LOW",
}

app = FastAPI(title="Dental X-Ray Inference API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[DentalInferencePipeline] = None


@app.on_event("startup")
async def _load_models() -> None:
    global pipeline
    print("[oral-xray] Loading two-stage pipeline...")
    pipeline = DentalInferencePipeline(
        yolo_weights=YOLO_WEIGHTS,
        classifier_weights=CLASSIFIER_WEIGHTS,
        device=DEVICE,
        yolo_conf=YOLO_CONF,
        classifier_conf_threshold=CLASSIFIER_REFINE_CONF,
        enable_refinement=True,
    )
    print("[oral-xray] Pipeline ready.")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy" if pipeline is not None else "loading",
        "model_loaded": pipeline is not None,
        "device": DEVICE,
        "yolo_classes": list(pipeline.yolo_class_names.values()) if pipeline else [],
        "classifier_classes": list(pipeline.classifier_class_names) if pipeline else [],
    }


def _normalize_to_rgb(raw: Image.Image) -> Image.Image:
    if raw.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.array(raw).astype(np.float32)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return raw.convert("RGB")


def _draw_annotations(image: Image.Image, detections: List[dict]) -> str:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    min_side = max(1, min(img_w, img_h))
    line_w = max(2, int(min_side * 0.003))
    font_size = max(12, int(min_side * 0.018))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        font = ImageFont.load_default()

    palette = [
        (255, 99, 99), (78, 205, 196), (69, 183, 209), (150, 206, 180),
        (255, 234, 167), (221, 160, 221), (152, 216, 200), (247, 220, 111),
        (187, 143, 206), (133, 193, 233),
    ]

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
        color = palette[idx % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        label = f"{det['class_name']} {det['confidence'] * 100:.0f}%"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 1), label, fill="white", font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def _serialize(result) -> List[dict]:
    out = []
    for d in result.detections:
        if d.refined_class_name and d.refined_confidence is not None and (
            d.was_refined or d.refined_confidence >= CLASSIFIER_REFINE_CONF
        ):
            cls = d.refined_class_name
            conf = float(d.refined_confidence)
        else:
            cls = d.class_name
            conf = float(d.confidence)

        x1, y1, x2, y2 = [float(v) for v in d.bbox]
        out.append({
            "class_name": cls,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "severity": SEVERITY_MAP.get(cls, "UNKNOWN"),
            "yolo_class": d.class_name,
            "yolo_confidence": float(d.confidence),
            "was_refined": bool(d.was_refined),
        })
    return out


@app.post("/predict_for_llm")
async def predict_for_llm(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    raw = Image.open(io.BytesIO(contents))
    image = _normalize_to_rgb(raw)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name, quality=95)
        tmp_path = tmp.name

    try:
        result = pipeline.predict(tmp_path)
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass

    detections = _serialize(result)
    annotated_b64 = None
    try:
        annotated_b64 = _draw_annotations(image, detections)
    except Exception as render_error:
        print(f"[oral-xray] render error: {render_error}")

    findings = {}
    for d in detections:
        findings.setdefault(d["class_name"], []).append(d["confidence"])
    summary_lines = []
    for cls, confs in sorted(findings.items(), key=lambda x: max(x[1]), reverse=True):
        summary_lines.append(f"{cls}: {len(confs)} occurrence(s), max conf {max(confs):.2f}")

    return {
        "success": True,
        "modality": "dental_xray",
        "num_detections": len(detections),
        "detections": detections,
        "annotated_image_base64": annotated_b64,
        "summary": "; ".join(summary_lines) if summary_lines else "No findings.",
        "timing_ms": round(result.total_time_ms, 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
