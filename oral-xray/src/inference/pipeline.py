"""
Production Inference Pipeline: YOLO Detection + ConvNeXt Classification
========================================================================

Pipeline:
1. YOLO detects objects in dental X-ray
2. Crop detected regions (+padding)
3. ConvNeXt reclassifies each crop
4. Refine label when ConvNeXt is confident
5. Return structured JSON + optional visualization

Features:
- Batched inference for efficiency
- Confidence-based label refinement
- Comprehensive error handling
- Logging and metrics
- Input validation
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    
    # ConvNeXt refinement
    refined_class_id: Optional[int] = None
    refined_class_name: Optional[str] = None
    refined_confidence: Optional[float] = None
    was_refined: bool = False
    
    # Metadata
    crop_size: Optional[Tuple[int, int]] = None


@dataclass
class InferenceResult:
    """Complete inference result for one image."""
    image_path: str
    image_size: Tuple[int, int]  # width, height
    detections: List[Detection] = field(default_factory=list)
    
    # Timing
    yolo_time_ms: float = 0.0
    classification_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Counts
    num_detections: int = 0
    num_refined: int = 0
    
    # Metadata
    timestamp: str = ""
    pipeline_version: str = "1.0.0"
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        # Convert detections to JSON-safe format
        safe_detections = []
        for d in self.detections:
            det_dict = {
                "class_id": int(d.class_id),
                "class_name": d.class_name,
                "confidence": float(d.confidence),
                "bbox": [float(x) for x in d.bbox],  # Convert numpy floats
                "refined_class_id": int(d.refined_class_id) if d.refined_class_id is not None else None,
                "refined_class_name": d.refined_class_name,
                "refined_confidence": float(d.refined_confidence) if d.refined_confidence is not None else None,
                "was_refined": d.was_refined,
                "crop_size": list(d.crop_size) if d.crop_size else None
            }
            safe_detections.append(det_dict)
        
        return {
            "image_path": self.image_path,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "detections": safe_detections,
            "timing": {
                "yolo_ms": round(self.yolo_time_ms, 2),
                "classification_ms": round(self.classification_time_ms, 2),
                "total_ms": round(self.total_time_ms, 2)
            },
            "counts": {
                "detections": self.num_detections,
                "refined": self.num_refined
            },
            "metadata": {
                "timestamp": self.timestamp,
                "pipeline_version": self.pipeline_version
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class DentalInferencePipeline:
    """
    Production inference pipeline combining YOLO detection and ConvNeXt classification.
    
    Usage:
        pipeline = DentalInferencePipeline(
            yolo_weights="path/to/yolo/best.pt",
            classifier_weights="path/to/convnext/best.pt",
            device="cuda:0"
        )
        result = pipeline.predict("dental_xray.jpg")
        print(result.to_json())
    """
    
    def __init__(
        self,
        yolo_weights: str,
        classifier_weights: str,
        device: str = "cuda:0",
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        classifier_conf_threshold: float = 0.85,
        crop_padding: float = 0.12,
        min_crop_size: int = 24,
        batch_size: int = 16,
        enable_refinement: bool = True
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            yolo_weights: Path to YOLO model weights
            classifier_weights: Path to ConvNeXt classifier weights
            device: Device for inference (cuda:0, cpu, etc.)
            yolo_conf: YOLO confidence threshold
            yolo_iou: YOLO IoU threshold for NMS
            classifier_conf_threshold: Minimum confidence to use classifier prediction
            crop_padding: Padding ratio around detected boxes (0.12 = 12%)
            min_crop_size: Minimum crop size in pixels
            batch_size: Batch size for classification
            enable_refinement: Whether to use ConvNeXt for label refinement
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Store config
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.classifier_conf_threshold = classifier_conf_threshold
        self.crop_padding = crop_padding
        self.min_crop_size = min_crop_size
        self.batch_size = batch_size
        self.enable_refinement = enable_refinement
        
        # Load models
        self._load_yolo(yolo_weights)
        if enable_refinement:
            self._load_classifier(classifier_weights)
        
        # Warmup
        self._warmup()
        
        logger.info("Pipeline initialized successfully")
    
    def _load_yolo(self, weights_path: str):
        """Load YOLO model."""
        logger.info(f"Loading YOLO model from: {weights_path}")
        self.yolo = YOLO(weights_path)
        self.yolo_class_names = self.yolo.names  # {0: 'class_name', ...}
        logger.info(f"YOLO classes: {list(self.yolo_class_names.values())}")
    
    def _load_classifier(self, weights_path: str):
        """Load ConvNeXt classifier."""
        logger.info(f"Loading classifier from: {weights_path}")
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Get model config from checkpoint
        self.classifier_class_names = checkpoint.get('class_names', [])
        model_name = checkpoint.get('model_name', 'convnext_large')
        input_size = checkpoint.get('input_size', 224)
        num_classes = len(self.classifier_class_names)
        
        # Import and create model
        import sys
        from pathlib import Path
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from src.models.classification import ClassificationModel, CLASSIFICATION_MODELS
        
        # Get full model name from registry
        model_cfg = CLASSIFICATION_MODELS.get(model_name, {})
        timm_name = model_cfg.get('name', model_name)
        
        self.classifier = ClassificationModel(
            model_name=timm_name,
            num_classes=num_classes,
            pretrained=False,
            dropout=0.0
        )
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        self.classifier_input_size = input_size
        
        # Create transforms
        from torchvision import transforms
        self.classifier_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        logger.info(f"Classifier loaded: {model_name}, {num_classes} classes, input size {input_size}")
        logger.info(f"Classifier classes: {self.classifier_class_names}")
    
    def _warmup(self):
        """Warmup models with dummy inference."""
        logger.info("Warming up models...")
        
        # YOLO warmup
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo.predict(dummy_img, verbose=False)
        
        # Classifier warmup
        if self.enable_refinement:
            dummy_tensor = torch.zeros(1, 3, self.classifier_input_size, self.classifier_input_size).to(self.device)
            with torch.no_grad():
                self.classifier(dummy_tensor)
        
        logger.info("Warmup complete")
    
    def _crop_detection(
        self, 
        image: Image.Image, 
        bbox: Tuple[float, float, float, float]
    ) -> Optional[Image.Image]:
        """
        Crop detection region with padding.
        
        Args:
            image: PIL Image
            bbox: (x1, y1, x2, y2) coordinates
            
        Returns:
            Cropped PIL Image or None if too small
        """
        W, H = image.size
        x1, y1, x2, y2 = bbox
        
        # Calculate box dimensions
        bw = x2 - x1
        bh = y2 - y1
        
        # Add padding
        if self.crop_padding > 0:
            px = bw * self.crop_padding
            py = bh * self.crop_padding
            x1 -= px
            y1 -= py
            x2 += px
            y2 += py
        
        # Clip to image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(W, int(x2))
        y2 = min(H, int(y2))
        
        # Check minimum size
        if (x2 - x1) < self.min_crop_size or (y2 - y1) < self.min_crop_size:
            return None
        
        return image.crop((x1, y1, x2, y2))
    
    def _classify_crops(self, crops: List[Image.Image]) -> List[Tuple[int, str, float]]:
        """
        Classify a batch of crops.
        
        Args:
            crops: List of PIL Images
            
        Returns:
            List of (class_id, class_name, confidence) tuples
        """
        if not crops:
            return []
        
        # Prepare batch
        tensors = [self.classifier_transform(crop) for crop in crops]
        batch = torch.stack(tensors).to(self.device)
        
        # Inference
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = self.classifier(batch)
                probs = F.softmax(outputs, dim=1)
        
        # Get predictions
        results = []
        for prob in probs:
            conf, pred_idx = prob.max(0)
            pred_idx = pred_idx.item()
            conf = conf.item()
            class_name = self.classifier_class_names[pred_idx]
            results.append((pred_idx, class_name, conf))
        
        return results
    
    def predict(
        self, 
        image_path: Union[str, Path],
        save_visualization: bool = False,
        output_dir: Optional[str] = None
    ) -> InferenceResult:
        """
        Run full inference pipeline on an image.
        
        Args:
            image_path: Path to input image
            save_visualization: Whether to save visualization
            output_dir: Directory for visualization output
            
        Returns:
            InferenceResult with all detections and metadata
        """
        start_time = time.time()
        image_path = Path(image_path)
        
        # Validate input
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        
        # Initialize result
        result = InferenceResult(
            image_path=str(image_path),
            image_size=(W, H),
            timestamp=datetime.now().isoformat()
        )
        
        # Step 1: YOLO Detection
        yolo_start = time.time()
        yolo_results = self.yolo.predict(
            image,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            verbose=False
        )[0]
        result.yolo_time_ms = (time.time() - yolo_start) * 1000
        
        # Process detections
        boxes = yolo_results.boxes
        detections = []
        crops = []
        crop_indices = []  # Track which detections have valid crops
        
        for i, box in enumerate(boxes):
            bbox = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = self.yolo_class_names[class_id]
            
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=tuple(bbox)
            )
            
            # Crop for classification
            if self.enable_refinement:
                crop = self._crop_detection(image, tuple(bbox))
                if crop is not None:
                    detection.crop_size = crop.size
                    crops.append(crop)
                    crop_indices.append(i)
            
            detections.append(detection)
        
        # Step 2: ConvNeXt Classification (batched)
        if self.enable_refinement and crops:
            cls_start = time.time()
            
            # Process in batches
            all_predictions = []
            for batch_start in range(0, len(crops), self.batch_size):
                batch_crops = crops[batch_start:batch_start + self.batch_size]
                batch_preds = self._classify_crops(batch_crops)
                all_predictions.extend(batch_preds)
            
            result.classification_time_ms = (time.time() - cls_start) * 1000
            
            # Step 3: Refine labels
            for crop_idx, (pred_idx, pred_name, pred_conf) in zip(crop_indices, all_predictions):
                det = detections[crop_idx]
                det.refined_class_id = pred_idx
                det.refined_class_name = pred_name
                det.refined_confidence = pred_conf
                
                # Refine if classifier is confident and disagrees
                if pred_conf >= self.classifier_conf_threshold:
                    if pred_name != det.class_name:
                        det.was_refined = True
                        result.num_refined += 1
                        logger.debug(f"Refined: {det.class_name} -> {pred_name} (conf={pred_conf:.3f})")
        
        # Finalize result
        result.detections = detections
        result.num_detections = len(detections)
        result.total_time_ms = (time.time() - start_time) * 1000
        
        # Visualization
        if save_visualization and output_dir:
            self._save_visualization(image, result, output_dir)
        
        return result
    
    def _save_visualization(
        self, 
        image: Image.Image, 
        result: InferenceResult,
        output_dir: str
    ):
        """Save visualization with bounding boxes and labels."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        draw = ImageDraw.Draw(image)
        
        # Color palette
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
        ]
        
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            
            # Use refined or original label
            if det.was_refined and det.refined_class_name:
                label = f"{det.refined_class_name} ({det.refined_confidence:.2f})*"
                class_id = det.refined_class_id
            else:
                label = f"{det.class_name} ({det.confidence:.2f})"
                class_id = det.class_id
            
            color = colors[class_id % len(colors)]
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label background
            text_bbox = draw.textbbox((x1, y1 - 20), label)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
            draw.text((x1, y1 - 20), label, fill="white")
        
        # Save
        stem = Path(result.image_path).stem
        output_path = output_dir / f"{stem}_inference.jpg"
        image.save(output_path, quality=95)
        logger.info(f"Visualization saved: {output_path}")
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        save_visualizations: bool = False,
        output_dir: Optional[str] = None
    ) -> List[InferenceResult]:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of image paths
            save_visualizations: Whether to save visualizations
            output_dir: Output directory
            
        Returns:
            List of InferenceResult objects
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path, save_visualizations, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue
        return results
    
    def get_final_predictions(self, result: InferenceResult) -> List[Dict]:
        """
        Get final predictions using refined labels when available.
        
        Returns list of dicts with final class assignments.
        """
        predictions = []
        for det in result.detections:
            if det.was_refined and det.refined_class_name:
                pred = {
                    "class": det.refined_class_name,
                    "confidence": det.refined_confidence,
                    "bbox": det.bbox,
                    "source": "classifier",
                    "yolo_class": det.class_name,
                    "yolo_confidence": det.confidence
                }
            else:
                pred = {
                    "class": det.class_name,
                    "confidence": det.confidence,
                    "bbox": det.bbox,
                    "source": "yolo"
                }
            predictions.append(pred)
        return predictions


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dental X-ray Inference Pipeline")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--yolo", required=True, help="YOLO weights path")
    parser.add_argument("--classifier", required=True, help="Classifier weights path")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--visualize", action="store_true", help="Save visualization")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--cls-conf", type=float, default=0.85, help="Classifier confidence for refinement")
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DentalInferencePipeline(
        yolo_weights=args.yolo,
        classifier_weights=args.classifier,
        device=args.device,
        yolo_conf=args.conf,
        classifier_conf_threshold=args.cls_conf
    )
    
    # Run inference
    result = pipeline.predict(
        args.image,
        save_visualization=args.visualize,
        output_dir=args.output
    )
    
    # Print results
    print("\n" + "="*60)
    print("INFERENCE RESULT")
    print("="*60)
    print(f"Image: {result.image_path}")
    print(f"Size: {result.image_size[0]}x{result.image_size[1]}")
    print(f"Detections: {result.num_detections}")
    print(f"Refined: {result.num_refined}")
    print(f"Time (YOLO): {result.yolo_time_ms:.1f}ms")
    print(f"Time (Classification): {result.classification_time_ms:.1f}ms")
    print(f"Time (Total): {result.total_time_ms:.1f}ms")
    print("\nDetections:")
    for i, det in enumerate(result.detections):
        if det.was_refined:
            print(f"  [{i}] {det.class_name} -> {det.refined_class_name} "
                  f"(YOLO: {det.confidence:.2f}, ConvNeXt: {det.refined_confidence:.2f}) *REFINED*")
        else:
            print(f"  [{i}] {det.class_name} ({det.confidence:.2f})")
    
    # Save JSON
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{Path(args.image).stem}_result.json"
    json_path.write_text(result.to_json())
    print(f"\nJSON saved: {json_path}")


if __name__ == "__main__":
    main()
