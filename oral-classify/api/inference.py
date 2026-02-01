"""
Inference pipeline for oral disease classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from torchvision import transforms, models

from .gradcam import get_gradcam_for_convnext


# Class names in order (must match training)
CLASS_NAMES = [
    "Calculus",
    "Caries",
    "Discoloration",
    "Gingivitis",
    "Hypodontia",
    "Ulcer"
]

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class OralClassifier:
    """
    Oral disease classifier with GradCAM visualization support.
    """
    
    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        num_classes: int = 6,
        model_type: str = "convnext_tiny"
    ):
        """
        Initialize the classifier.
        
        Args:
            weights_path: Path to model weights
            device: Device to run on
            num_classes: Number of classes
            model_type: Model architecture (convnext_tiny, resnet50)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_names = CLASS_NAMES
        self.model_type = model_type
        
        # Load model
        self.model = self._build_model(model_type, num_classes)
        self._load_weights(weights_path)
        self.model.eval()
        
        # Initialize GradCAM
        self.gradcam = get_gradcam_for_convnext(self.model, use_plus_plus=True)
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        
        print(f"Model loaded on {self.device}")
        print(f"Classes: {self.class_names}")
    
    def _build_model(self, model_type: str, num_classes: int) -> nn.Module:
        """Build the model architecture."""
        if model_type == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            model = models.convnext_tiny(weights=weights)
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes)
            )
        elif model_type == "convnext_small":
            # SOTA model architecture
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            model = models.convnext_small(weights=weights)
            in_features = model.classifier[2].in_features
            # Simple linear head as used in SOTA training
            model.classifier[2] = nn.Linear(in_features, num_classes)
        elif model_type == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def _load_weights(self, weights_path: str):
        """Load model weights."""
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Loaded weights from: {weights_path}")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(
        self, 
        image: Image.Image,
        top_k: int = 3
    ) -> Dict:
        """
        Predict oral disease class.
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            Dict with prediction, confidence, and top-k results
        """
        input_tensor = self.preprocess(image)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = probs.topk(top_k)
        
        results = {
            "predicted_class": self.class_names[top_indices[0].item()],
            "confidence": top_probs[0].item(),
            "top_k": [
                {
                    "class": self.class_names[idx.item()],
                    "probability": prob.item()
                }
                for prob, idx in zip(top_probs, top_indices)
            ],
            "all_probabilities": {
                name: probs[i].item() 
                for i, name in enumerate(self.class_names)
            }
        }
        
        return results
    
    def predict_with_gradcam(
        self,
        image: Image.Image,
        target_class: int = None,
        alpha: float = 0.5
    ) -> Tuple[Dict, np.ndarray]:
        """
        Predict with GradCAM visualization.
        
        Args:
            image: PIL Image
            target_class: Class to visualize (None = predicted class)
            alpha: Heatmap opacity
            
        Returns:
            Tuple of (prediction dict, overlay image as numpy array)
        """
        input_tensor = self.preprocess(image)
        
        # Get prediction first
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        predicted_idx = probs.argmax().item()
        
        # Generate GradCAM
        original_np = np.array(image.resize((224, 224)))
        if original_np.max() <= 1.0:
            original_np = (original_np * 255).astype(np.uint8)
        
        if target_class is None:
            target_class = predicted_idx
        
        overlay = self.gradcam.generate_overlay(
            input_tensor, 
            original_np, 
            target_class=target_class,
            alpha=alpha
        )
        
        results = {
            "predicted_class": self.class_names[predicted_idx],
            "confidence": probs[predicted_idx].item(),
            "gradcam_class": self.class_names[target_class],
            "all_probabilities": {
                name: probs[i].item() 
                for i, name in enumerate(self.class_names)
            }
        }
        
        return results, overlay
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "classes": self.class_names,
            "device": str(self.device),
            "input_size": 224
        }


# Singleton instance for API
_classifier_instance: Optional[OralClassifier] = None


def get_classifier(
    weights_path: str = None,
    device: str = "cuda",
    model_type: str = "convnext_small"  # Default to SOTA model
) -> OralClassifier:
    """
    Get or create classifier singleton.
    
    Uses default SOTA weights if not specified.
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        # Default to SOTA weights
        if weights_path is None:
            weights_path = str(
                Path(__file__).parent.parent / 
                "SOTA_FINAL_20251124_1300" / "best_model.pth"
            )
        
        _classifier_instance = OralClassifier(
            weights_path=weights_path,
            device=device,
            model_type=model_type
        )
    
    return _classifier_instance
