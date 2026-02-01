"""
GradCAM visualization for interpretable predictions.
Shows which regions of the image the model focuses on for classification.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    
    Works with ConvNeXt and other CNN architectures.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: The neural network model
            target_layer: The layer to compute GradCAM on (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self, 
        input_tensor: torch.Tensor, 
        target_class: int = None
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap for the input.
        
        Args:
            input_tensor: Preprocessed input image [1, C, H, W]
            target_class: Class to generate heatmap for (None = predicted class)
            
        Returns:
            Heatmap as numpy array [H, W] with values in [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_overlay(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: int = None,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap overlaid on the original image.
        
        Args:
            input_tensor: Preprocessed input image [1, C, H, W]
            original_image: Original image as numpy array [H, W, C] (RGB, 0-255)
            target_class: Class to generate heatmap for
            alpha: Opacity of the heatmap overlay
            colormap: OpenCV colormap to use
            
        Returns:
            Overlaid image as numpy array [H, W, C] (RGB, 0-255)
        """
        # Generate heatmap
        cam = self.generate(input_tensor, target_class)
        
        # Resize heatmap to match original image
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = (original_image * (1 - alpha) + heatmap * alpha).astype(np.uint8)
        
        return overlay


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks.
    
    Better at localizing multiple instances and partial visibility.
    """
    
    def generate(
        self, 
        input_tensor: torch.Tensor, 
        target_class: int = None
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # Grad-CAM++ weights (2nd and 3rd order gradients approximation)
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        # Avoid division by zero
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        denominator = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        
        # Alpha weights
        alpha = grad_2 / denominator
        
        # Weighted gradients (ReLU of gradients * alpha)
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam


def get_gradcam_for_convnext(model: torch.nn.Module, use_plus_plus: bool = True):
    """
    Create GradCAM for ConvNeXt model.
    
    Args:
        model: ConvNeXt model
        use_plus_plus: Use Grad-CAM++ (better localization)
        
    Returns:
        GradCAM or GradCAMPlusPlus instance
    """
    # For ConvNeXt, target the last layer before classifier
    # features[-1] is the last stage; features[-1][-1] is the last block
    try:
        target_layer = model.features[-1][-1]
    except (AttributeError, IndexError):
        # Fallback for different model structures
        target_layer = list(model.children())[-2]
    
    if use_plus_plus:
        return GradCAMPlusPlus(model, target_layer)
    else:
        return GradCAM(model, target_layer)
