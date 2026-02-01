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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1
    )
