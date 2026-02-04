"""
Oral Disease Detection API

FastAPI server for YOLO-based oral disease detection.
Returns JSON diagnosis suitable for LLM report generation.

Endpoints:
    POST /predict - Detect oral diseases and return JSON diagnosis
    POST /predict_text - Return plain text diagnosis for LLM
    POST /predict_for_llm - LLM-optimized output for report generation

Classes detected:
    - Caries: Tooth decay/cavities
    - Ulcer: Oral lesions
    - Tooth Discoloration: Tooth color changes
    - Gingivitis: Gum inflammation
"""

import io
import base64
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import cv2


# ==================== CONFIG ====================
MODEL_PATH = Path(__file__).parent.parent / "oral_runs/YOLOv11x_ORAL_SOTA_20251125_0709/weights/last.pt"
CLASS_NAMES = ["Caries", "Ulcer", "Tooth Discoloration", "Gingivitis"]
CONFIDENCE_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Clinical descriptions
CLASS_DESCRIPTIONS = {
    "Caries": "dental caries (tooth decay/cavities) requiring restorative treatment - may progress to pulpitis if untreated",
    "Ulcer": "oral ulcer/lesion requiring clinical evaluation - monitor for healing, consider biopsy if persistent > 2 weeks",
    "Tooth Discoloration": "tooth discoloration that may indicate trauma, decay, or systemic conditions",
    "Gingivitis": "gingival inflammation indicating early periodontal disease - reversible with proper treatment",
}

SEVERITY_MAP = {
    "Caries": "HIGH",
    "Ulcer": "HIGH",
    "Tooth Discoloration": "MODERATE",
    "Gingivitis": "MODERATE",
}


# ==================== MODELS ====================
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]
    description: str
    severity: str


class DiagnosisResult(BaseModel):
    success: bool
    timestamp: str
    image_size: List[int]
    num_detections: int
    detections: List[Detection]
    summary: str
    findings: List[str]
    recommendations: List[str]


class TextDiagnosis(BaseModel):
    success: bool
    diagnosis_text: str


# ==================== APP ====================
app = FastAPI(
    title="Oral Disease Detection API",
    description="YOLO-based oral disease detection with LLM-ready output",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


def load_model():
    """Load YOLO model."""
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
        model.to(DEVICE)
        print(f"✓ Model loaded from {MODEL_PATH}")
    return model


def process_image(file_bytes: bytes) -> Image.Image:
    """Process uploaded image."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def run_detection(image: Image.Image) -> tuple:
    """Run YOLO detection."""
    model = load_model()
    img_array = np.array(image)
    
    results = model(img_array, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    
    detections = []
    findings = []
    
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        
        detection = Detection(
            class_name=class_name,
            confidence=round(confidence, 4),
            bbox=[round(b, 1) for b in bbox],
            description=CLASS_DESCRIPTIONS.get(class_name, "Unknown finding"),
            severity=SEVERITY_MAP.get(class_name, "UNKNOWN"),
        )
        detections.append(detection)
        
        finding = f"{class_name.upper()} detected ({confidence:.1%}) - {CLASS_DESCRIPTIONS.get(class_name, '')}"
        findings.append(finding)
    
    return detections, findings, image.size


def generate_summary(detections: List[Detection]) -> str:
    """Generate summary text."""
    if not detections:
        return "No significant oral abnormalities detected."
    
    counts = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
    
    parts = [f"{count} {cls}" for cls, count in counts.items()]
    return f"Detected: {', '.join(parts)}."


def generate_recommendations(detections: List[Detection]) -> List[str]:
    """Generate clinical recommendations."""
    recommendations = []
    
    has_caries = any(d.class_name == "Caries" for d in detections)
    has_ulcer = any(d.class_name == "Ulcer" for d in detections)
    has_gingivitis = any(d.class_name == "Gingivitis" for d in detections)
    has_discolor = any(d.class_name == "Tooth Discoloration" for d in detections)
    
    if has_caries:
        recommendations.append("URGENT: Dental caries detected - schedule restorative dental appointment")
        recommendations.append("Radiographic evaluation recommended to assess caries depth")
        recommendations.append("Consider fluoride treatment and dietary counseling")
    
    if has_ulcer:
        recommendations.append("Oral ulcer present - monitor for healing over 2 weeks")
        recommendations.append("If persistent, biopsy recommended to rule out malignancy")
        recommendations.append("Evaluate for systemic causes if recurrent")
    
    if has_gingivitis:
        recommendations.append("Gingivitis detected - professional dental cleaning recommended")
        recommendations.append("Improved oral hygiene instruction needed")
        recommendations.append("Follow-up in 4-6 weeks to assess treatment response")
    
    if has_discolor:
        recommendations.append("Tooth discoloration noted - determine if intrinsic or extrinsic")
        recommendations.append("Vitality testing recommended for affected teeth")
    
    if not detections:
        recommendations.append("No abnormalities detected - continue routine dental care")
        recommendations.append("Regular dental check-ups every 6 months recommended")
    
    return recommendations


# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    return {
        "name": "Oral Disease Detection API",
        "version": "1.0.0",
        "model": str(MODEL_PATH.name),
        "classes": CLASS_NAMES,
        "device": DEVICE,
    }


@app.get("/health")
async def health():
    try:
        load_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/predict", response_model=DiagnosisResult)
async def predict(file: UploadFile = File(...)):
    """Detect oral diseases and return structured JSON."""
    contents = await file.read()
    image = process_image(contents)
    
    detections, findings, img_size = run_detection(image)
    summary = generate_summary(detections)
    recommendations = generate_recommendations(detections)
    
    return DiagnosisResult(
        success=True,
        timestamp=datetime.now().isoformat(),
        image_size=list(img_size),
        num_detections=len(detections),
        detections=detections,
        summary=summary,
        findings=findings,
        recommendations=recommendations,
    )


@app.post("/predict_text", response_model=TextDiagnosis)
async def predict_text(file: UploadFile = File(...)):
    """Return plain text diagnosis for LLM input."""
    contents = await file.read()
    image = process_image(contents)
    
    detections, findings, img_size = run_detection(image)
    summary = generate_summary(detections)
    recommendations = generate_recommendations(detections)
    
    text = "# ORAL DISEASE AI DETECTION REPORT\n\n"
    text += f"Timestamp: {datetime.now().isoformat()}\n"
    text += f"Image: {img_size[0]}x{img_size[1]} pixels\n\n"
    
    text += "## SUMMARY\n"
    text += f"{summary}\n\n"
    
    text += "## DETAILED FINDINGS\n"
    if findings:
        for i, finding in enumerate(findings, 1):
            text += f"{i}. {finding}\n"
    else:
        text += "No significant abnormalities detected.\n"
    
    text += "\n## RECOMMENDATIONS\n"
    for i, rec in enumerate(recommendations, 1):
        text += f"{i}. {rec}\n"
    
    text += "\n---\n"
    text += "Note: AI analysis should be reviewed by a qualified dental professional.\n"
    
    return TextDiagnosis(success=True, diagnosis_text=text)


@app.post("/predict_for_llm")
async def predict_for_llm(file: UploadFile = File(...)):
    """Return LLM-optimized JSON for report generation."""
    contents = await file.read()
    image = process_image(contents)
    
    detections, findings, img_size = run_detection(image)
    
    return {
        "patient_context": "Oral/dental examination image analysis",
        "modality": "Intraoral photograph",
        "body_part": "Oral cavity",
        "ai_findings": [
            {
                "finding": d.class_name,
                "location": f"bbox {d.bbox}",
                "confidence": f"{d.confidence:.1%}",
                "severity": d.severity,
                "clinical_meaning": CLASS_DESCRIPTIONS.get(d.class_name, "Unknown"),
            }
            for d in detections
        ],
        "conditions_detected": {
            "caries": any(d.class_name == "Caries" for d in detections),
            "ulcer": any(d.class_name == "Ulcer" for d in detections),
            "gingivitis": any(d.class_name == "Gingivitis" for d in detections),
            "discoloration": any(d.class_name == "Tooth Discoloration" for d in detections),
        },
        "urgency": "HIGH" if any(d.severity == "HIGH" for d in detections) else "ROUTINE",
        "summary": generate_summary(detections),
        "recommendations": generate_recommendations(detections),
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Oral Disease Detection API...")
    uvicorn.run(app, host="0.0.0.0", port=8002)
