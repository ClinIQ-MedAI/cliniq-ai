"""
Classification models for dental X-ray analysis.
"""

import timm
import torch
import torch.nn as nn
from typing import Dict, Optional


class ClassificationModel(nn.Module):
    """
    General classification model wrapper with transfer learning support.
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.5,
        multi_label: bool = False
    ):

        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.multi_label = multi_label
        
        # Create model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[-1]
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        print(f"Created {model_name} with {num_classes} classes")
        print(f"Feature dimension: {feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if self.multi_label:
            return torch.sigmoid(logits)
        else:
            return logits
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_classification_model(config: Dict) -> nn.Module:

    model_config = config.get('model', {})
    
    model = ClassificationModel(
        model_name=model_config.get('name', 'resnet50'),
        num_classes=model_config.get('num_classes', 5),
        pretrained=model_config.get('pretrained', True),
        dropout=model_config.get('dropout', 0.5),
        multi_label=model_config.get('multi_label', False)
    )
    
    # Freeze backbone if specified
    if model_config.get('freeze_backbone', False):
        model.freeze_backbone()
        print("Backbone frozen for initial training")
    
    return model


# Predefined model configurations
# Ordered by recommended priority for dental X-ray classification with A6000 GPU

CLASSIFICATION_MODELS = {
    # ============ TOP TIER - Best for Medical Imaging ============
    # ConvNeXt family - excellent for medical imaging, good inductive bias
    'convnext_large': {'name': 'convnext_large.fb_in22k_ft_in1k', 'input_size': 224},
    'convnext_xlarge': {'name': 'convnext_xlarge.fb_in22k_ft_in1k', 'input_size': 224},  # 350M params
    'convnext_base': {'name': 'convnext_base.fb_in22k_ft_in1k', 'input_size': 224},
    
    # EfficientNetV2 - excellent speed/accuracy tradeoff
    'efficientnetv2_l': {'name': 'tf_efficientnetv2_l.in21k_ft_in1k', 'input_size': 384},
    'efficientnetv2_m': {'name': 'tf_efficientnetv2_m.in21k_ft_in1k', 'input_size': 384},
    'efficientnetv2_s': {'name': 'tf_efficientnetv2_s.in21k_ft_in1k', 'input_size': 384},
    
    # MaxViT - hybrid CNN+Transformer, excellent for varied crop sizes
    'maxvit_base': {'name': 'maxvit_base_tf_384.in21k_ft_in1k', 'input_size': 384},
    'maxvit_large': {'name': 'maxvit_large_tf_384.in21k_ft_in1k', 'input_size': 384},
    
    # EVA-02 - SOTA vision model, excellent for medical imaging
    'eva02_base': {'name': 'eva02_base_patch14_448.mim_in22k_ft_in1k', 'input_size': 448},
    'eva02_large': {'name': 'eva02_large_patch14_448.mim_in22k_ft_in1k', 'input_size': 448},
    
    # ============ MEDIUM TIER - Good balance ============
    # Swin Transformer V2 - robust to input variations
    'swinv2_base': {'name': 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k', 'input_size': 256},
    'swinv2_large': {'name': 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k', 'input_size': 256},
    
    # CaiT - class-attention in image transformers
    'cait_s36': {'name': 'cait_s36_384.fb_dist_in1k', 'input_size': 384},
    
    # DeiT III - robust transformer baseline
    'deit3_base': {'name': 'deit3_base_patch16_384.fb_in22k_ft_in1k', 'input_size': 384},
    'deit3_large': {'name': 'deit3_large_patch16_384.fb_in22k_ft_in1k', 'input_size': 384},
    
    # BEiT - BERT-style pre-training for vision
    'beit_base': {'name': 'beit_base_patch16_384.in22k_ft_in22k_in1k', 'input_size': 384},
    'beit_large': {'name': 'beit_large_patch16_384.in22k_ft_in22k_in1k', 'input_size': 384},
    
    # ============ LIGHTWEIGHT - Fast inference ============
    'efficientnet_b4': {'name': 'efficientnet_b4.ra2_in1k', 'input_size': 380},
    'efficientnet_b5': {'name': 'efficientnet_b5.sw_in12k_ft_in1k', 'input_size': 448},
    'convnext_small': {'name': 'convnext_small.fb_in22k_ft_in1k', 'input_size': 224},
    'convnext_tiny': {'name': 'convnext_tiny.fb_in22k_ft_in1k', 'input_size': 224},
    
    # ============ LEGACY - For comparison ============
    'resnet50': {'name': 'resnet50.a1_in1k', 'input_size': 224},
    'resnet101': {'name': 'resnet101.a1h_in1k', 'input_size': 224},
    'resnet152': {'name': 'resnet152.a1h_in1k', 'input_size': 224},
    'vit_base_patch16_224': {'name': 'vit_base_patch16_224.augreg_in21k_ft_in1k', 'input_size': 224},
    'vit_large_patch16_224': {'name': 'vit_large_patch16_224.augreg_in21k_ft_in1k', 'input_size': 224},
}


# ============ RECOMMENDED CONFIGS FOR DENTAL X-RAY ============
DENTAL_RECOMMENDED_MODELS = {
    # Best overall accuracy (if you have time to train)
    'best_accuracy': 'convnext_xlarge',
    
    # Best for your A6000 GPU (good balance of speed and accuracy)
    'recommended': 'convnext_large',
    
    # Best for small objects (Apical Periodontitis, Decay)
    'small_objects': 'eva02_base',
    
    # Fastest inference while maintaining accuracy
    'fast_inference': 'efficientnetv2_m',
    
    # Best for limited data (strong regularization)
    'limited_data': 'maxvit_base',
}

