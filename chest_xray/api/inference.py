"""
Chest X-ray Classification Inference Module
Supports GradCAM visualization for explainability.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Dict, List, Tuple, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.convnext import ConvNeXtClassifier

# Disease classes (13 classes, excluding Hernia)
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Infiltration", 
    "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# Default checkpoint path
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "outputs/checkpoints/best.pt"


class ChestXrayClassifier:
    """Chest X-ray multi-label classifier with GradCAM support."""
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or str(DEFAULT_CHECKPOINT)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Image transforms (matching training)
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # GradCAM target layer
        self.target_layer = self.model.get_cam_target_layer()
        
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        model = ConvNeXtClassifier(
            num_classes=len(CLASS_NAMES),
            pretrained=False,
            dropout=0.2
        )
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "ema_state_dict" in checkpoint:
            state_dict = checkpoint["ema_state_dict"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        
        return model
    
    def predict(
        self, 
        image: Image.Image,
        threshold: float = 0.5,
        top_k: int = 5
    ) -> Dict:
        """
        Predict diseases from chest X-ray.
        
        Returns:
            Dict with predictions, probabilities, and detected conditions.
        """
        # Preprocess
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Build results
        all_probs = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}
        
        # Detected conditions (above threshold)
        detected = [
            {"class": cls, "probability": float(probs[i])}
            for i, cls in enumerate(CLASS_NAMES)
            if probs[i] >= threshold
        ]
        detected.sort(key=lambda x: -x["probability"])
        
        # Top-K predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_results = [
            {"class": CLASS_NAMES[i], "probability": float(probs[i])}
            for i in top_indices
        ]
        
        # Primary prediction (highest probability)
        primary_idx = int(np.argmax(probs))
        
        return {
            "primary_prediction": CLASS_NAMES[primary_idx],
            "primary_confidence": float(probs[primary_idx]),
            "detected_conditions": detected,
            "top_k": top_k_results,
            "all_probabilities": all_probs,
            "threshold_used": threshold
        }
    
    def predict_with_gradcam(
        self,
        image: Image.Image,
        target_class: Optional[int] = None,
        alpha: float = 0.5
    ) -> Tuple[Dict, np.ndarray]:
        """
        Predict with GradCAM visualization.
        
        Returns:
            Tuple of (prediction_dict, overlay_image_array)
        """
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Keep original for overlay
        original_np = np.array(image.resize((512, 512))) / 255.0
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        primary_idx = int(np.argmax(probs))
        target_idx = target_class if target_class is not None else primary_idx
        
        # GradCAM
        def reshape_transform(tensor, height=16, width=16):
            # ConvNeXt output is (B, C, H, W), no reshape needed
            return tensor
        
        cam = GradCAMPlusPlus(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=None
        )
        
        # Multi-label targets
        class MultiLabelTarget:
            def __init__(self, category):
                self.category = category
            def __call__(self, model_output):
                return model_output[:, self.category]
        
        grayscale_cam = cam(
            input_tensor=tensor,
            targets=[MultiLabelTarget(target_idx)]
        )[0]
        
        # Create overlay
        overlay = show_cam_on_image(
            original_np.astype(np.float32),
            grayscale_cam,
            use_rgb=True,
            colormap=2  # JET
        )
        
        all_probs = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}
        
        result = {
            "primary_prediction": CLASS_NAMES[primary_idx],
            "primary_confidence": float(probs[primary_idx]),
            "gradcam_class": CLASS_NAMES[target_idx],
            "all_probabilities": all_probs
        }
        
        return result, overlay
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "model_type": "ConvNeXt-Large",
            "num_classes": len(CLASS_NAMES),
            "classes": CLASS_NAMES,
            "device": str(self.device),
            "input_size": 512
        }


# Global classifier instance
_classifier = None


def get_classifier() -> ChestXrayClassifier:
    """Get or create classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = ChestXrayClassifier()
    return _classifier


if __name__ == "__main__":
    # Test
    print("Loading classifier...")
    clf = get_classifier()
    print(f"Model info: {clf.get_model_info()}")
