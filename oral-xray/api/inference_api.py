"""
FastAPI inference server for dental X-ray models.

Usage:
    uvicorn api.inference_api:app --host 0.0.0.0 --port 8000 --reload

Example request:
    curl -X POST http://localhost:8000/predict \\
        -F "file=@test_image.jpg"
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import io
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ===== Configuration =====
MODEL_PATH = "outputs/exports/model.onnx"  # Update with your model path
IMAGE_SIZE = 224
NUM_CLASSES = 5
CLASS_NAMES = {
    0: "Caries",
    1: "Impacted Tooth",
    2: "Periapical Lesion",
    3: "Deep Caries",
    4: "Crown"
}  # Update with your class names


# ===== Pydantic Models =====
class PredictionResponse(BaseModel):
    """Response model for prediction."""
    predicted_class: int
    class_name: str
    confidence: float
    all_probabilities: Dict[str, float]
    processing_time: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    model_path: str


# ===== FastAPI App =====
app = FastAPI(
    title="ClinIQ Oral X-Ray Inference API",
    description="API for dental X-ray classification and detection",
    version="0.1.0"
)


# ===== Global Variables =====
ort_session: Optional[ort.InferenceSession] = None
transform = None


# ===== Startup/Shutdown Events =====
@app.on_event("startup")
async def load_model():
    """Load ONNX model on startup."""
    global ort_session, transform
    
    try:
        print(f"Loading ONNX model from {MODEL_PATH}...")
        ort_session = ort.InferenceSession(MODEL_PATH)
        
        # Create transform
        transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        print("✓ Model loaded successfully")
        print(f"  Input: {ort_session.get_inputs()[0].name}, shape: {ort_session.get_inputs()[0].shape}")
        print(f"  Output: {ort_session.get_outputs()[0].name}, shape: {ort_session.get_outputs()[0].shape}")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("  API will start but predictions will fail.")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global ort_session
    if ort_session:
        del ort_session
    print("✓ Model unloaded")


# ===== Utility Functions =====
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess uploaded image for inference.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Preprocessed image tensor
    """
    # Read image
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image.convert('RGB'))
    
    # Apply transforms
    transformed = transform(image=image)
    img_tensor = transformed['image'].numpy()
    
    # Add batch dimension
    img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32)
    
    return img_tensor


def softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax to logits."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ===== API Endpoints =====
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "ClinIQ Oral X-Ray Inference API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if ort_session is not None else "unhealthy",
        model_loaded=ort_session is not None,
        model_path=MODEL_PATH
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict dental condition from X-ray image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction with class, confidence, and probabilities
    """
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        start_time = time.time()
        
        # Read and preprocess image
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)
        
        # Run inference
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]  # Remove batch dimension
        
        # Apply softmax
        probs = softmax(logits)
        
        # Get prediction
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        
        # Create probability dictionary
        all_probs = {
            CLASS_NAMES.get(i, f"Class_{i}"): float(probs[i])
            for i in range(len(probs))
        }
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            predicted_class=pred_class,
            class_name=CLASS_NAMES.get(pred_class, f"Class_{pred_class}"),
            confidence=confidence,
            all_probabilities=all_probs,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Batch prediction on multiple images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of predictions
    """
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images allowed")
    
    try:
        start_time = time.time()
        predictions = []
        
        for file in files:
            # Read and preprocess
            image_bytes = await file.read()
            input_tensor = preprocess_image(image_bytes)
            
            # Inference
            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: input_tensor})
            logits = outputs[0][0]
            
            # Process results
            probs = softmax(logits)
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            
            predictions.append({
                "filename": file.filename,
                "predicted_class": pred_class,
                "class_name": CLASS_NAMES.get(pred_class, f"Class_{pred_class}"),
                "confidence": confidence
            })
        
        processing_time = time.time() - start_time
        
        return {
            "predictions": predictions,
            "total_images": len(files),
            "total_processing_time": processing_time,
            "avg_time_per_image": processing_time / len(files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ===== Main =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
