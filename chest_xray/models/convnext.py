"""
ConvNeXt-Large Model for Multi-Label Chest X-ray Classification
- Sigmoid activation (NOT softmax)
- ImageNet pretrained
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Large classifier for multi-label chest X-ray classification.
    
    Features:
    - ImageNet pretrained backbone
    - Sigmoid outputs (not softmax)
    - Dropout for regularization
    """
    
    def __init__(
        self,
        num_classes: int = 13,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Load pretrained ConvNeXt-Large
        weights = models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_large(weights=weights)
        
        # Get feature dimension (1536 for ConvNeXt-Large)
        num_features = 1536
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(num_features),
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
            # NO softmax - using BCEWithLogitsLoss or FocalLoss
        )
        
        # Initialize final layer
        nn.init.xavier_uniform_(self.backbone.classifier[3].weight)
        nn.init.zeros_(self.backbone.classifier[3].bias)
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
        
        Returns:
            Logits of shape (batch, num_classes) - NOT probabilities!
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier (for Grad-CAM)."""
        # ConvNeXt uses features as feature extractor
        features = self.backbone.features(x)
        return features
    
    def get_cam_target_layer(self):
        """Get target layer for Grad-CAM."""
        # Last block of the feature extractor (Stage 3 -> Last Block)
        # torchvision ConvNeXt features[-1] is the last Sequential (Stage 3)
        # We target the last block within that stage for best resolution
        return self.backbone.features[-1][-1]


class EMA:
    """
    Exponential Moving Average of model weights.
    
    Maintains a shadow copy of model weights that gets updated
    with exponential moving average during training.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def create_model(
    model_name: str = "convnext_large",
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """
    Create model based on name.
    
    Args:
        model_name: Model name (convnext_large, convnext_base, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
    
    Returns:
        Model instance
    """
    if model_name == "convnext_large":
        return ConvNeXtClassifier(num_classes, pretrained, dropout)
    elif model_name == "convnext_base":
        return ConvNeXtBaseClassifier(num_classes, pretrained, dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class ConvNeXtBaseClassifier(nn.Module):
    """ConvNeXt-Base classifier (smaller, faster)."""
    
    def __init__(
        self,
        num_classes: int = 13,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_base(weights=weights)
        
        num_features = 1024  # ConvNeXt-Base
        
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(num_features),
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )
        
        nn.init.xavier_uniform_(self.backbone.classifier[3].weight)
        nn.init.zeros_(self.backbone.classifier[3].bias)
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_cam_target_layer(self):
        return self.backbone.features[-1]


if __name__ == "__main__":
    # Test model
    model = ConvNeXtClassifier(num_classes=13, pretrained=True)
    print(f"Model: ConvNeXt-Large")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output (logits): {out[0, :5]}")  # First 5 logits
    print(f"Output (probs): {torch.sigmoid(out[0, :5])}")  # First 5 probabilities
