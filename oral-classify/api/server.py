"""
FastAPI server for oral disease classification with GradCAM visualization.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload

Or:
    python -m api.server
"""

import io
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

from api.inference import get_classifier, CLASS_NAMES


# ============ API Models ============
class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    top_k: List[dict]
    all_probabilities: dict


class GradCAMResult(BaseModel):
    predicted_class: str
    confidence: float
    gradcam_class: str
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
    title="Oral Disease Classification API",
    description="AI-powered oral disease classification with GradCAM visualization",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier
classifier = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global classifier
    print("🔧 Loading oral disease classifier...")
    
    try:
        classifier = get_classifier()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    if classifier is None:
        return HealthResponse(
            status="error",
            model_loaded=False,
            model_type="unknown",
            num_classes=0,
            classes=[],
            device="none"
        )
    
    info = classifier.get_model_info()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        **info
    )


@app.get("/classes")
async def get_classes():
    """Get available disease classes."""
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=6)
):
    """
    Classify uploaded oral image.
    
    Returns predicted disease class and probabilities.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Predict
        result = classifier.predict(image, top_k=top_k)
        
        return PredictionResult(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/gradcam")
async def predict_with_gradcam(
    file: UploadFile = File(...),
    target_class: Optional[str] = Query(None, description="Class to visualize (default: predicted)"),
    alpha: float = Query(0.5, ge=0.1, le=0.9, description="Heatmap opacity")
):
    """
    Classify image and return GradCAM visualization.
    
    Shows which regions the model focuses on for the prediction.
    Returns the overlaid image as JPEG.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get target class index if specified
        target_idx = None
        if target_class:
            if target_class in CLASS_NAMES:
                target_idx = CLASS_NAMES.index(target_class)
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unknown class: {target_class}. Available: {CLASS_NAMES}"
                )
        
        # Predict with GradCAM
        result, overlay = classifier.predict_with_gradcam(
            image, 
            target_class=target_idx,
            alpha=alpha
        )
        
        # Convert overlay to bytes
        overlay_img = Image.fromarray(overlay)
        img_bytes = io.BytesIO()
        overlay_img.save(img_bytes, format="JPEG", quality=95)
        img_bytes.seek(0)
        
        return StreamingResponse(
            img_bytes,
            media_type="image/jpeg",
            headers={
                "X-Predicted-Class": result["predicted_class"],
                "X-Confidence": str(round(result["confidence"], 4)),
                "X-GradCAM-Class": result["gradcam_class"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/gradcam/json")
async def predict_with_gradcam_json(
    file: UploadFile = File(...),
    target_class: Optional[str] = Query(None),
    alpha: float = Query(0.5, ge=0.1, le=0.9)
):
    """
    Classify image with GradCAM and return JSON with base64 encoded image.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        import base64
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get target class index
        target_idx = None
        if target_class and target_class in CLASS_NAMES:
            target_idx = CLASS_NAMES.index(target_class)
        
        # Predict with GradCAM
        result, overlay = classifier.predict_with_gradcam(
            image, 
            target_class=target_idx,
            alpha=alpha
        )
        
        # Encode overlay as base64
        overlay_img = Image.fromarray(overlay)
        img_bytes = io.BytesIO()
        overlay_img.save(img_bytes, format="JPEG", quality=90)
        img_bytes.seek(0)
        b64_image = base64.b64encode(img_bytes.read()).decode()
        
        return {
            **result,
            "gradcam_image_base64": b64_image,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ LLM-Ready Endpoints ============

# Clinical descriptions for LLM
CLASS_DESCRIPTIONS = {
    "Calculus": "dental calculus (tarite) accumulation requiring professional cleaning",
    "Caries": "dental caries (tooth decay/cavities) requiring restorative treatment",
    "Discoloration": "tooth discoloration that may indicate underlying dental issues",
    "Gingivitis": "gingival inflammation indicating early periodontal disease",
    "Hypodontia": "congenitally missing teeth requiring orthodontic evaluation",
    "Ulcer": "oral ulcer/lesion requiring clinical evaluation and possible biopsy",
}

SEVERITY_MAP = {
    "Calculus": "MODERATE",
    "Caries": "HIGH",
    "Discoloration": "LOW",
    "Gingivitis": "MODERATE",
    "Hypodontia": "INFO",
    "Ulcer": "HIGH",
}


@app.post("/predict_for_llm")
async def predict_for_llm(file: UploadFile = File(...)):
    """
    Classify oral image and return LLM-optimized JSON for report generation.
    
    Includes clinical context, findings, and recommendations.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = classifier.predict(image, top_k=6)
        
        predicted = result["predicted_class"]
        confidence = result["confidence"]
        
        # Generate recommendations
        recommendations = []
        if predicted == "Caries":
            recommendations = [
                "Recommend dental examination for caries assessment",
                "Consider radiographic evaluation to determine extent",
                "Treatment options: filling, crown, or root canal depending on severity",
            ]
        elif predicted == "Calculus":
            recommendations = [
                "Professional dental cleaning (scaling) recommended",
                "Oral hygiene instruction and follow-up",
                "Evaluate for underlying periodontal disease",
            ]
        elif predicted == "Gingivitis":
            recommendations = [
                "Professional dental cleaning recommended",
                "Improved oral hygiene practices advised",
                "Follow-up to assess treatment response",
                "Consider periodontal evaluation if persistent",
            ]
        elif predicted == "Ulcer":
            recommendations = [
                "Clinical examination to determine ulcer etiology",
                "Consider biopsy if ulcer persists >2 weeks",
                "Evaluate for systemic conditions if recurrent",
                "Symptomatic treatment with topical agents",
            ]
        elif predicted == "Hypodontia":
            recommendations = [
                "Orthodontic consultation recommended",
                "Consider prosthetic replacement options",
                "Genetic counseling may be appropriate",
            ]
        elif predicted == "Discoloration":
            recommendations = [
                "Determine cause (intrinsic vs extrinsic)",
                "Dental cleaning for extrinsic stains",
                "Evaluate for pulp vitality if single tooth involved",
            ]
        
        # Build findings
        findings = [
            f"{predicted.upper()} detected with {confidence:.1%} confidence",
            f"Clinical significance: {CLASS_DESCRIPTIONS.get(predicted, 'Unknown condition')}",
        ]
        
        # Top differentials
        if len(result["top_k"]) > 1:
            differentials = [f"{r['class']} ({r['probability']:.1%})" for r in result["top_k"][1:3]]
            findings.append(f"Differential considerations: {', '.join(differentials)}")
        
        return {
            "patient_context": "Dental/oral examination image analysis",
            "modality": "Intraoral photograph",
            "body_part": "Oral cavity",
            "ai_findings": {
                "primary_diagnosis": predicted,
                "confidence": f"{confidence:.1%}",
                "severity": SEVERITY_MAP.get(predicted, "UNKNOWN"),
                "clinical_meaning": CLASS_DESCRIPTIONS.get(predicted, "Unknown"),
            },
            "differential_diagnoses": [
                {"condition": r["class"], "probability": f"{r['probability']:.1%}"}
                for r in result["top_k"][1:4]
            ],
            "all_probabilities": result["all_probabilities"],
            "urgency": "HIGH" if SEVERITY_MAP.get(predicted) == "HIGH" else "ROUTINE",
            "findings": findings,
            "recommendations": recommendations,
            "summary": f"AI analysis detected {predicted.lower()} with {confidence:.1%} confidence. {CLASS_DESCRIPTIONS.get(predicted, '')}",
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_text")
async def predict_text(file: UploadFile = File(...)):
    """
    Classify oral image and return plain text diagnosis for LLM input.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = classifier.predict(image, top_k=6)
        
        predicted = result["predicted_class"]
        confidence = result["confidence"]
        
        text = "# ORAL DISEASE AI ANALYSIS REPORT\n\n"
        text += f"Timestamp: {datetime.now().isoformat()}\n\n"
        
        text += "## PRIMARY DIAGNOSIS\n"
        text += f"Condition: {predicted}\n"
        text += f"Confidence: {confidence:.1%}\n"
        text += f"Severity: {SEVERITY_MAP.get(predicted, 'UNKNOWN')}\n"
        text += f"Description: {CLASS_DESCRIPTIONS.get(predicted, 'Unknown condition')}\n\n"
        
        text += "## DIFFERENTIAL DIAGNOSES\n"
        for i, r in enumerate(result["top_k"][:4], 1):
            text += f"{i}. {r['class']}: {r['probability']:.1%}\n"
        
        text += "\n## ALL PROBABILITY SCORES\n"
        for cls, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            text += f"  {cls}: {prob:.1%} {bar}\n"
        
        text += "\n---\n"
        text += "Note: This AI analysis should be reviewed by a qualified dental professional.\n"
        
        return {"success": True, "diagnosis_text": text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1
    )

