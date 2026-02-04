"""
Bone X-ray Detection API

FastAPI server for YOLO-based pediatric wrist trauma detection.
Returns JSON diagnosis suitable for LLM report generation.

Endpoints:
    POST /predict - Detect anomalies and return JSON diagnosis
    POST /predict_text - Return plain text diagnosis for LLM

Classes detected:
    - fracture: Bone fractures
    - metal: Surgical hardware
    - periostealreaction: Bone healing indicator
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
MODEL_PATH = Path(__file__).parent.parent / "outputs/YOLO11x_TOP3_20260203_0645/weights/best.pt"
CLASS_NAMES = ["fracture", "metal", "periostealreaction"]
CONFIDENCE_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Clinical descriptions for LLM
CLASS_DESCRIPTIONS = {
    "fracture": "bone fracture requiring immediate medical attention",
    "metal": "surgical metal hardware/implant from previous intervention",
    "periostealreaction": "periosteal reaction indicating bone healing or pathological process",
}

SEVERITY_MAP = {
    "fracture": "HIGH",
    "metal": "INFO",
    "periostealreaction": "MODERATE",
}


# ==================== MODELS ====================
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    description: str
    severity: str


class DiagnosisResult(BaseModel):
    success: bool
    timestamp: str
    image_size: List[int]  # [width, height]
    num_detections: int
    detections: List[Detection]
    summary: str
    findings: List[str]
    recommendations: List[str]
    llm_prompt: str  # Pre-formatted prompt for LLM


class TextDiagnosis(BaseModel):
    success: bool
    diagnosis_text: str


# ==================== APP ====================
app = FastAPI(
    title="Bone X-ray Detection API",
    description="YOLO-based pediatric wrist trauma detection with LLM-ready output",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
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
        print(f"✓ Device: {DEVICE}")
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
    """Run YOLO detection on image."""
    model = load_model()
    
    # Convert to numpy
    img_array = np.array(image)
    
    # Run inference
    results = model(img_array, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    
    detections = []
    findings = []
    
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = CLASS_NAMES[class_id]
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
        
        # Create finding text
        finding = f"{class_name.upper()} detected (confidence: {confidence:.1%}) - {CLASS_DESCRIPTIONS[class_name]}"
        findings.append(finding)
    
    return detections, findings, image.size


def generate_summary(detections: List[Detection]) -> str:
    """Generate summary text."""
    if not detections:
        return "No significant abnormalities detected in the X-ray image."
    
    counts = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
    
    parts = []
    for cls, count in counts.items():
        if count == 1:
            parts.append(f"1 {cls}")
        else:
            parts.append(f"{count} {cls}s")
    
    return f"Detected: {', '.join(parts)}."


def generate_recommendations(detections: List[Detection]) -> List[str]:
    """Generate clinical recommendations."""
    recommendations = []
    
    has_fracture = any(d.class_name == "fracture" for d in detections)
    has_metal = any(d.class_name == "metal" for d in detections)
    has_periosteal = any(d.class_name == "periostealreaction" for d in detections)
    
    if has_fracture:
        recommendations.append("URGENT: Fracture detected - recommend immediate orthopedic consultation")
        recommendations.append("Consider immobilization and pain management")
        recommendations.append("Additional imaging may be required to assess fracture extent")
    
    if has_periosteal:
        recommendations.append("Periosteal reaction present - may indicate healing fracture or underlying pathology")
        recommendations.append("Recommend clinical correlation and follow-up imaging")
    
    if has_metal:
        recommendations.append("Previously implanted surgical hardware detected")
        recommendations.append("Review prior surgical history for context")
    
    if not detections:
        recommendations.append("No abnormalities detected - clinical correlation recommended")
        recommendations.append("Consider additional views if symptoms persist")
    
    return recommendations


def generate_llm_prompt(detections: List[Detection], findings: List[str], recommendations: List[str]) -> str:
    """Generate prompt for LLM report generation."""
    prompt = """You are a radiologist writing a clinical report for a pediatric wrist X-ray.

## AI Detection Results:
"""
    
    if detections:
        prompt += "\n### Findings:\n"
        for finding in findings:
            prompt += f"- {finding}\n"
        
        prompt += "\n### Detection Details:\n"
        for d in detections:
            prompt += f"- **{d.class_name}** (confidence: {d.confidence:.1%}, severity: {d.severity})\n"
            prompt += f"  Location: bbox [{d.bbox[0]:.0f}, {d.bbox[1]:.0f}, {d.bbox[2]:.0f}, {d.bbox[3]:.0f}]\n"
    else:
        prompt += "\nNo significant abnormalities detected by AI analysis.\n"
    
    prompt += "\n### Recommended Actions:\n"
    for rec in recommendations:
        prompt += f"- {rec}\n"
    
    prompt += """
## Instructions:
Write a professional radiology report based on these AI findings. Include:
1. Clinical indication (assume wrist pain/trauma)
2. Technique description
3. Findings section with anatomical descriptions
4. Impression with numbered conclusions
5. Recommendations for follow-up if needed

Use formal medical terminology. Maintain a professional, clinical tone.
"""
    
    return prompt


# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Bone X-ray Detection API",
        "version": "1.0.0",
        "model": str(MODEL_PATH),
        "classes": CLASS_NAMES,
        "device": DEVICE,
    }


@app.get("/health")
async def health():
    """Health check."""
    try:
        load_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/predict", response_model=DiagnosisResult)
async def predict(file: UploadFile = File(...)):
    """
    Detect bone abnormalities and return structured JSON diagnosis.
    
    Returns complete diagnosis with:
    - Detection boxes and classes
    - Clinical findings
    - Recommendations
    - LLM-ready prompt for report generation
    """
    contents = await file.read()
    image = process_image(contents)
    
    detections, findings, img_size = run_detection(image)
    summary = generate_summary(detections)
    recommendations = generate_recommendations(detections)
    llm_prompt = generate_llm_prompt(detections, findings, recommendations)
    
    return DiagnosisResult(
        success=True,
        timestamp=datetime.now().isoformat(),
        image_size=list(img_size),
        num_detections=len(detections),
        detections=detections,
        summary=summary,
        findings=findings,
        recommendations=recommendations,
        llm_prompt=llm_prompt,
    )


@app.post("/predict_text", response_model=TextDiagnosis)
async def predict_text(file: UploadFile = File(...)):
    """
    Detect bone abnormalities and return plain text diagnosis.
    
    Perfect for passing directly to an LLM for report generation.
    """
    contents = await file.read()
    image = process_image(contents)
    
    detections, findings, img_size = run_detection(image)
    summary = generate_summary(detections)
    recommendations = generate_recommendations(detections)
    
    # Build text
    text = "# BONE X-RAY AI ANALYSIS REPORT\n\n"
    text += f"Timestamp: {datetime.now().isoformat()}\n"
    text += f"Image dimensions: {img_size[0]}x{img_size[1]} pixels\n\n"
    
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
    text += "Note: This AI analysis should be reviewed by a qualified radiologist.\n"
    
    return TextDiagnosis(
        success=True,
        diagnosis_text=text,
    )


@app.post("/predict_for_llm")
async def predict_for_llm(file: UploadFile = File(...)):
    """
    Return detection results formatted specifically for LLM consumption.
    
    Returns a JSON object with all the context an LLM needs to write a report.
    """
    contents = await file.read()
    image = process_image(contents)
    
    detections, findings, img_size = run_detection(image)
    
    # Simplified output for LLM
    return {
        "patient_context": "Pediatric patient presenting with wrist pain/trauma",
        "modality": "X-ray",
        "body_part": "Wrist",
        "ai_findings": [
            {
                "finding": d.class_name,
                "confidence": f"{d.confidence:.1%}",
                "severity": d.severity,
                "clinical_meaning": CLASS_DESCRIPTIONS[d.class_name],
            }
            for d in detections
        ],
        "has_fracture": any(d.class_name == "fracture" for d in detections),
        "has_hardware": any(d.class_name == "metal" for d in detections),
        "has_healing_signs": any(d.class_name == "periostealreaction" for d in detections),
        "urgency": "HIGH" if any(d.severity == "HIGH" for d in detections) else "ROUTINE",
        "summary": generate_summary(detections),
        "recommendations": generate_recommendations(detections),
    }


# ==================== ARABIC ENDPOINTS ====================

# Arabic translations
CLASS_NAMES_AR = {
    "fracture": "كسر",
    "metal": "معدن جراحي",
    "periostealreaction": "تفاعل السمحاق",
}

CLASS_DESCRIPTIONS_AR = {
    "fracture": "كسر عظمي يتطلب عناية طبية فورية",
    "metal": "أجهزة معدنية جراحية من تدخل سابق",
    "periostealreaction": "تفاعل السمحاق يشير إلى التئام العظام أو عملية مرضية",
}

SEVERITY_AR = {
    "HIGH": "عالي",
    "MODERATE": "متوسط",
    "INFO": "معلومات",
}


def generate_summary_ar(detections: List[Detection]) -> str:
    """Generate Arabic summary."""
    if not detections:
        return "لم يتم اكتشاف أي تشوهات مهمة في صورة الأشعة السينية."
    
    counts = {}
    for d in detections:
        ar_name = CLASS_NAMES_AR.get(d.class_name, d.class_name)
        counts[ar_name] = counts.get(ar_name, 0) + 1
    
    parts = [f"{count} {cls}" for cls, count in counts.items()]
    return f"تم اكتشاف: {', '.join(parts)}."


def generate_recommendations_ar(detections: List[Detection]) -> List[str]:
    """Generate Arabic recommendations."""
    recommendations = []
    
    has_fracture = any(d.class_name == "fracture" for d in detections)
    has_metal = any(d.class_name == "metal" for d in detections)
    has_periosteal = any(d.class_name == "periostealreaction" for d in detections)
    
    if has_fracture:
        recommendations.append("عاجل: تم اكتشاف كسر - يوصى باستشارة جراحة العظام فوراً")
        recommendations.append("فكر في التثبيت وإدارة الألم")
        recommendations.append("قد تكون هناك حاجة لتصوير إضافي لتقييم مدى الكسر")
    
    if has_periosteal:
        recommendations.append("يوجد تفاعل السمحاق - قد يشير إلى كسر في طور الالتئام أو حالة مرضية")
        recommendations.append("يوصى بالربط السريري والمتابعة بالتصوير")
    
    if has_metal:
        recommendations.append("تم اكتشاف أجهزة جراحية مزروعة سابقاً")
        recommendations.append("مراجعة التاريخ الجراحي السابق للسياق")
    
    if not detections:
        recommendations.append("لم يتم اكتشاف تشوهات - يوصى بالربط السريري")
        recommendations.append("فكر في إجراء تصوير إضافي إذا استمرت الأعراض")
    
    return recommendations


@app.post("/predict_text_ar")
async def predict_text_arabic(file: UploadFile = File(...)):
    """
    تقرير التشخيص بالعربية
    Arabic diagnosis report for LLM.
    """
    contents = await file.read()
    image = process_image(contents)
    
    detections, findings, img_size = run_detection(image)
    summary = generate_summary_ar(detections)
    recommendations = generate_recommendations_ar(detections)
    
    text = "# تقرير تحليل الأشعة السينية للعظام\n\n"
    text += f"التاريخ: {datetime.now().isoformat()}\n"
    text += f"أبعاد الصورة: {img_size[0]}x{img_size[1]} بكسل\n\n"
    
    text += "## الملخص\n"
    text += f"{summary}\n\n"
    
    text += "## النتائج التفصيلية\n"
    if detections:
        for i, d in enumerate(detections, 1):
            ar_name = CLASS_NAMES_AR.get(d.class_name, d.class_name)
            ar_desc = CLASS_DESCRIPTIONS_AR.get(d.class_name, "")
            text += f"{i}. {ar_name} (الثقة: {d.confidence:.1%}) - {ar_desc}\n"
    else:
        text += "لم يتم اكتشاف أي تشوهات مهمة.\n"
    
    text += "\n## التوصيات\n"
    for i, rec in enumerate(recommendations, 1):
        text += f"{i}. {rec}\n"
    
    text += "\n---\n"
    text += "ملاحظة: يجب مراجعة هذا التحليل من قبل أخصائي أشعة مؤهل.\n"
    
    return {"success": True, "diagnosis_text": text, "language": "ar"}


@app.post("/predict_for_llm_ar")
async def predict_for_llm_arabic(file: UploadFile = File(...)):
    """
    نتائج الكشف بصيغة JSON للذكاء الاصطناعي
    Arabic LLM-optimized JSON for report generation.
    """
    contents = await file.read()
    image = process_image(contents)
    
    detections, findings, img_size = run_detection(image)
    
    return {
        "language": "ar",
        "patient_context": "مريض أطفال يعاني من ألم/إصابة في المعصم",
        "modality": "أشعة سينية",
        "body_part": "المعصم",
        "ai_findings": [
            {
                "finding": CLASS_NAMES_AR.get(d.class_name, d.class_name),
                "finding_en": d.class_name,
                "confidence": f"{d.confidence:.1%}",
                "severity": SEVERITY_AR.get(d.severity, d.severity),
                "clinical_meaning": CLASS_DESCRIPTIONS_AR.get(d.class_name, ""),
            }
            for d in detections
        ],
        "has_fracture": any(d.class_name == "fracture" for d in detections),
        "urgency": "عالي" if any(d.severity == "HIGH" for d in detections) else "روتيني",
        "summary": generate_summary_ar(detections),
        "recommendations": generate_recommendations_ar(detections),
    }


# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    print("Starting Bone X-ray Detection API...")
    uvicorn.run(app, host="0.0.0.0", port=8001)

