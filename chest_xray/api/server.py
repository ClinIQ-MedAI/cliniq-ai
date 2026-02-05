"""
FastAPI server for NIH Chest X-ray Classification with GradCAM.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8003 --reload

Or:
    python -m api.server
"""

import io
import sys
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

from api.inference import get_classifier, CLASS_NAMES


# ============ Disease Metadata ============
CLASS_DESCRIPTIONS = {
    "Atelectasis": "Partial or complete lung collapse",
    "Cardiomegaly": "Enlarged heart, may indicate heart disease",
    "Consolidation": "Lung tissue filled with fluid/pus (infection)",
    "Edema": "Fluid accumulation in lungs (pulmonary edema)",
    "Effusion": "Fluid around the lungs (pleural effusion)",
    "Emphysema": "Chronic lung damage, often from smoking",
    "Fibrosis": "Lung scarring, reduced breathing capacity",
    "Infiltration": "Abnormal substance in lung tissue",
    "Mass": "Abnormal growth, requires further evaluation",
    "Nodule": "Small abnormal spot, may need follow-up",
    "Pleural_Thickening": "Thickened pleura, often from prior disease",
    "Pneumonia": "Lung infection requiring treatment",
    "Pneumothorax": "Collapsed lung from air leak (URGENT)",
}

CLASS_DESCRIPTIONS_AR = {
    "Atelectasis": "انهيار جزئي أو كلي للرئة",
    "Cardiomegaly": "تضخم القلب، قد يشير إلى مرض قلبي",
    "Consolidation": "امتلاء أنسجة الرئة بالسوائل (عدوى)",
    "Edema": "تراكم السوائل في الرئتين (وذمة رئوية)",
    "Effusion": "سائل حول الرئتين (انصباب جنبي)",
    "Emphysema": "تلف رئوي مزمن، غالباً من التدخين",
    "Fibrosis": "تليف الرئة، انخفاض قدرة التنفس",
    "Infiltration": "مادة غير طبيعية في أنسجة الرئة",
    "Mass": "نمو غير طبيعي، يتطلب تقييم إضافي",
    "Nodule": "بقعة صغيرة غير طبيعية، قد تحتاج متابعة",
    "Pleural_Thickening": "سماكة غشاء الجنب",
    "Pneumonia": "التهاب رئوي يتطلب علاج",
    "Pneumothorax": "انهيار الرئة من تسرب الهواء (عاجل)",
}

CLASS_NAMES_AR = {
    "Atelectasis": "انخماص الرئة",
    "Cardiomegaly": "تضخم القلب",
    "Consolidation": "تصلب رئوي",
    "Edema": "وذمة رئوية",
    "Effusion": "انصباب جنبي",
    "Emphysema": "انتفاخ الرئة",
    "Fibrosis": "تليف رئوي",
    "Infiltration": "ارتشاح رئوي",
    "Mass": "كتلة رئوية",
    "Nodule": "عقيدة رئوية",
    "Pleural_Thickening": "سماكة غشاء الجنب",
    "Pneumonia": "التهاب رئوي",
    "Pneumothorax": "استرواح صدري",
}

SEVERITY_MAP = {
    "Pneumothorax": "CRITICAL",
    "Pneumonia": "HIGH",
    "Mass": "HIGH",
    "Consolidation": "HIGH",
    "Edema": "HIGH",
    "Effusion": "MODERATE",
    "Cardiomegaly": "MODERATE",
    "Infiltration": "MODERATE",
    "Atelectasis": "MODERATE",
    "Nodule": "LOW",
    "Fibrosis": "LOW",
    "Emphysema": "LOW",
    "Pleural_Thickening": "LOW",
}


# ============ API Models ============
class PredictionResult(BaseModel):
    primary_prediction: str
    primary_confidence: float
    detected_conditions: List[dict]
    top_k: List[dict]
    all_probabilities: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    num_classes: int
    classes: List[str]
    device: str


# ============ App Setup ============
app = FastAPI(
    title="NIH Chest X-ray Classification API",
    description="AI-powered chest X-ray analysis with 13 disease detection and GradCAM visualization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = None


@app.on_event("startup")
async def load_model():
    global classifier
    print("🔧 Loading Chest X-ray classifier...")
    try:
        classifier = get_classifier()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if classifier is None:
        return HealthResponse(
            status="error", model_loaded=False, model_type="unknown",
            num_classes=0, classes=[], device="none"
        )
    info = classifier.get_model_info()
    return HealthResponse(status="healthy", model_loaded=True, **info)


@app.get("/classes")
async def get_classes():
    return {"classes": CLASS_NAMES, "num_classes": len(CLASS_NAMES)}


@app.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.1, le=0.9),
    top_k: int = Query(5, ge=1, le=13)
):
    """Classify chest X-ray image."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = classifier.predict(image, threshold=threshold, top_k=top_k)
        return PredictionResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/gradcam")
async def predict_with_gradcam(
    file: UploadFile = File(...),
    target_class: Optional[str] = Query(None),
    alpha: float = Query(0.5, ge=0.1, le=0.9)
):
    """Classify with GradCAM visualization. Returns JPEG image."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        target_idx = None
        if target_class and target_class in CLASS_NAMES:
            target_idx = CLASS_NAMES.index(target_class)
        
        result, overlay = classifier.predict_with_gradcam(image, target_class=target_idx, alpha=alpha)
        
        overlay_img = Image.fromarray(overlay)
        img_bytes = io.BytesIO()
        overlay_img.save(img_bytes, format="JPEG", quality=95)
        img_bytes.seek(0)
        
        return StreamingResponse(
            img_bytes,
            media_type="image/jpeg",
            headers={
                "X-Primary-Prediction": result["primary_prediction"],
                "X-Confidence": str(round(result["primary_confidence"], 4)),
                "X-GradCAM-Class": result["gradcam_class"]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ LLM-Ready Endpoints ============

@app.post("/predict_for_llm")
async def predict_for_llm(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(True, description="Include base64 GradCAM image")
):
    """
    Classify chest X-ray and return LLM-optimized JSON.
    Optionally includes base64 encoded GradCAM visualization.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get predictions with GradCAM
        result, overlay = classifier.predict_with_gradcam(image)
        
        # Encode GradCAM as base64
        gradcam_b64 = None
        if include_gradcam:
            overlay_img = Image.fromarray(overlay)
            img_bytes = io.BytesIO()
            overlay_img.save(img_bytes, format="JPEG", quality=85)
            img_bytes.seek(0)
            gradcam_b64 = base64.b64encode(img_bytes.read()).decode()
        
        primary = result["primary_prediction"]
        confidence = result["primary_confidence"]
        
        # Build findings
        findings = [
            f"{primary.upper()} detected with {confidence:.1%} confidence",
            f"Clinical significance: {CLASS_DESCRIPTIONS.get(primary, 'Unknown')}",
        ]
        
        # Detected conditions (above threshold)
        pred_result = classifier.predict(image, threshold=0.5)
        detected = pred_result["detected_conditions"]
        
        if len(detected) > 1:
            additional = [f"{d['class']} ({d['probability']:.1%})" for d in detected[1:3]]
            findings.append(f"Additional findings: {', '.join(additional)}")
        
        # Recommendations based on severity
        recommendations = []
        severity = SEVERITY_MAP.get(primary, "UNKNOWN")
        
        if severity == "CRITICAL":
            recommendations = [
                "URGENT: Immediate medical attention required",
                "Consider emergency intervention",
                "Notify attending physician immediately",
            ]
        elif severity == "HIGH":
            recommendations = [
                "Prompt medical evaluation recommended",
                "Consider additional imaging (CT scan)",
                "Clinical correlation advised",
            ]
        else:
            recommendations = [
                "Follow-up imaging may be beneficial",
                "Clinical correlation recommended",
                "Monitor for symptom progression",
            ]
        
        response = {
            "patient_context": "Chest X-ray radiograph analysis",
            "modality": "Chest X-ray (PA/AP view)",
            "body_part": "Thorax",
            "ai_findings": {
                "primary_diagnosis": primary,
                "confidence": f"{confidence:.1%}",
                "severity": severity,
                "clinical_meaning": CLASS_DESCRIPTIONS.get(primary, "Unknown"),
            },
            "detected_conditions": [
                {"condition": d["class"], "probability": f"{d['probability']:.1%}"}
                for d in detected
            ],
            "differential_diagnoses": [
                {"condition": r["class"], "probability": f"{r['probability']:.1%}"}
                for r in pred_result["top_k"][1:4]
            ],
            "all_probabilities": result["all_probabilities"],
            "urgency": "CRITICAL" if severity == "CRITICAL" else ("HIGH" if severity == "HIGH" else "ROUTINE"),
            "findings": findings,
            "recommendations": recommendations,
            "summary": f"AI analysis detected {primary.lower()} with {confidence:.1%} confidence. {CLASS_DESCRIPTIONS.get(primary, '')}",
            "timestamp": datetime.now().isoformat(),
        }
        
        if gradcam_b64:
            response["gradcam_image_base64"] = gradcam_b64
            response["gradcam_info"] = "Base64 encoded JPEG showing model attention regions"
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_for_llm_ar")
async def predict_for_llm_arabic(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(True)
):
    """تقرير تحليل أشعة الصدر بالعربية مع صورة GradCAM"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        result, overlay = classifier.predict_with_gradcam(image)
        
        gradcam_b64 = None
        if include_gradcam:
            overlay_img = Image.fromarray(overlay)
            img_bytes = io.BytesIO()
            overlay_img.save(img_bytes, format="JPEG", quality=85)
            img_bytes.seek(0)
            gradcam_b64 = base64.b64encode(img_bytes.read()).decode()
        
        primary = result["primary_prediction"]
        confidence = result["primary_confidence"]
        ar_name = CLASS_NAMES_AR.get(primary, primary)
        
        severity = SEVERITY_MAP.get(primary, "UNKNOWN")
        severity_ar = {
            "CRITICAL": "حرج",
            "HIGH": "مرتفع", 
            "MODERATE": "متوسط",
            "LOW": "منخفض"
        }.get(severity, "غير معروف")
        
        pred_result = classifier.predict(image, threshold=0.5)
        detected = pred_result["detected_conditions"]
        
        recommendations_ar = []
        if severity == "CRITICAL":
            recommendations_ar = [
                "عاجل: يتطلب تدخل طبي فوري",
                "فكر في التدخل الطارئ",
                "أبلغ الطبيب المعالج فوراً",
            ]
        elif severity == "HIGH":
            recommendations_ar = [
                "يوصى بتقييم طبي سريع",
                "فكر في تصوير إضافي (أشعة مقطعية)",
                "ينصح بالربط السريري",
            ]
        else:
            recommendations_ar = [
                "قد يكون التصوير المتابع مفيداً",
                "يوصى بالربط السريري",
                "راقب تطور الأعراض",
            ]
        
        response = {
            "language": "ar",
            "patient_context": "تحليل أشعة الصدر السينية",
            "modality": "أشعة صدر (منظر أمامي/خلفي)",
            "body_part": "الصدر",
            "ai_findings": {
                "primary_diagnosis": ar_name,
                "primary_diagnosis_en": primary,
                "confidence": f"{confidence:.1%}",
                "severity": severity_ar,
                "clinical_meaning": CLASS_DESCRIPTIONS_AR.get(primary, "غير معروف"),
            },
            "detected_conditions": [
                {
                    "condition": CLASS_NAMES_AR.get(d["class"], d["class"]),
                    "condition_en": d["class"],
                    "probability": f"{d['probability']:.1%}"
                }
                for d in detected
            ],
            "urgency": "حرج" if severity == "CRITICAL" else ("مرتفع" if severity == "HIGH" else "روتيني"),
            "recommendations": recommendations_ar,
            "summary": f"اكتشف التحليل {ar_name} بثقة {confidence:.1%}. {CLASS_DESCRIPTIONS_AR.get(primary, '')}",
            "timestamp": datetime.now().isoformat(),
        }
        
        if gradcam_b64:
            response["gradcam_image_base64"] = gradcam_b64
            response["gradcam_info"] = "صورة JPEG مشفرة base64 تظهر مناطق اهتمام النموذج"
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8003, reload=False, workers=1)
