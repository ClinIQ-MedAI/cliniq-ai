"""
DenseNet-121 for Multi-Label Chest X-ray Classification (CheXNet Architecture)
"""

import torch
import torch.nn as nn
from torchvision import models


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 based classifier for multi-label chest X-ray classification.
    Based on CheXNet architecture.
    
    Args:
        num_classes: Number of disease classes
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate before final layer
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Load pretrained DenseNet-121
        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
        )
        
        # Get number of features from backbone
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )
        
        # Initialize final layer
        nn.init.xavier_uniform_(self.backbone.classifier[1].weight)
        nn.init.zeros_(self.backbone.classifier[1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier for visualization."""
        features = self.backbone.features(x)
        features = nn.functional.relu(features, inplace=True)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return features.view(features.size(0), -1)
