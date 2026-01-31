"""
FastAPI server for dental X-ray inference.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Or for development:
    python api_server.py
"""

import io
import sys
import base64
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import uvicorn

from src.inference.pipeline import DentalInferencePipeline


# ============ Configuration ============
YOLO_WEIGHTS = str(PROJECT_ROOT / "outputs/experiments/yolo_v8x_base_1024/weights/best.pt")
CLASSIFIER_WEIGHTS = str(PROJECT_ROOT / "outputs/classification/convnext_large_20260130_090637/weights/best.pt")
DEVICE = "cuda:0"

# ============ API Models ============
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]
    refined_class_name: Optional[str] = None
    refined_confidence: Optional[float] = None
    was_refined: bool = False

class InferenceResponse(BaseModel):
    success: bool
    num_detections: int
    num_refined: int
    detections: List[Detection]
    timing_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str
    yolo_classes: List[str]
    classifier_classes: List[str]

# ============ App Setup ============
app = FastAPI(
    title="Dental X-Ray Analysis API",
    description="YOLO detection + ConvNeXt classification pipeline for dental X-ray analysis",
    version="1.0.0"
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[DentalInferencePipeline] = None


@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global pipeline
    print("🔧 Loading inference pipeline...")
    
    try:
        pipeline = DentalInferencePipeline(
            yolo_weights=YOLO_WEIGHTS,
            classifier_weights=CLASSIFIER_WEIGHTS,
            device=DEVICE,
            yolo_conf=0.25,
            classifier_conf_threshold=0.85,
            enable_refinement=True
        )
        print("✅ Pipeline loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    if pipeline is None:
        return HealthResponse(
            status="error",
            models_loaded=False,
            device="none",
            yolo_classes=[],
            classifier_classes=[]
        )
    
    return HealthResponse(
        status="healthy",
        models_loaded=True,
        device=str(pipeline.device),
        yolo_classes=list(pipeline.yolo_class_names.values()),
        classifier_classes=pipeline.classifier_class_names
    )


@app.post("/predict", response_model=InferenceResponse)
async def predict(
    file: UploadFile = File(...),
    yolo_conf: float = Query(0.25, ge=0.0, le=1.0, description="YOLO confidence threshold"),
    cls_conf: float = Query(0.85, ge=0.0, le=1.0, description="Classifier confidence for refinement")
):
    """
    Run inference on uploaded dental X-ray image.
    
    Returns detections with optional ConvNeXt refinement.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Save temporarily (pipeline expects path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        # Update thresholds
        pipeline.yolo_conf = yolo_conf
        pipeline.classifier_conf_threshold = cls_conf
        
        # Run inference
        result = pipeline.predict(tmp_path)
        
        # Clean up
        Path(tmp_path).unlink()
        
        # Format response
        detections = []
        for det in result.detections:
            detections.append(Detection(
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=list(det.bbox),
                refined_class_name=det.refined_class_name,
                refined_confidence=det.refined_confidence,
                was_refined=det.was_refined
            ))
        
        return InferenceResponse(
            success=True,
            num_detections=result.num_detections,
            num_refined=result.num_refined,
            detections=detections,
            timing_ms=result.total_time_ms,
            timestamp=result.timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/visualize")
async def predict_with_visualization(
    file: UploadFile = File(...),
    yolo_conf: float = Query(0.25, ge=0.0, le=1.0),
    cls_conf: float = Query(0.85, ge=0.0, le=1.0)
):
    """
    Run inference and return visualization image with bounding boxes.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Save temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        # Update thresholds
        pipeline.yolo_conf = yolo_conf
        pipeline.classifier_conf_threshold = cls_conf
        
        # Run inference with visualization
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = pipeline.predict(tmp_path, save_visualization=True, output_dir=tmp_dir)
            
            # Find visualization
            vis_path = Path(tmp_dir) / f"{Path(tmp_path).stem}_inference.jpg"
            if vis_path.exists():
                vis_image = Image.open(vis_path)
                
                # Convert to bytes
                img_bytes = io.BytesIO()
                vis_image.save(img_bytes, format="JPEG", quality=95)
                img_bytes.seek(0)
                
                return StreamingResponse(
                    img_bytes,
                    media_type="image/jpeg",
                    headers={
                        "X-Num-Detections": str(result.num_detections),
                        "X-Num-Refined": str(result.num_refined),
                        "X-Timing-Ms": str(round(result.total_time_ms, 2))
                    }
                )
        
        # Cleanup
        Path(tmp_path).unlink()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
async def get_classes():
    """Get available detection classes."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "yolo_classes": list(pipeline.yolo_class_names.values()),
        "classifier_classes": pipeline.classifier_class_names
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
