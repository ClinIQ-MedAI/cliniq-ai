"""
Grad-CAM Visualization for Chest X-ray Classification
Generates heatmaps to explain model predictions
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pathlib import Path


class GradCAM:
    """
    Grad-CAM implementation for ConvNeXt models.
    
    Generates class activation maps to visualize which regions
    the model focuses on for each disease prediction.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The trained model
            target_layer: The layer to compute CAM for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.handles = []  # Track hooks to remove them later
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Remove hooks to prevent memory leaks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int
    ) -> np.ndarray:
        """
        Generate Grad-CAM for a specific class.
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            class_idx: Target class index
        
        Returns:
            CAM heatmap as numpy array (H, W)
        """
        self.model.eval()
        
        # Forward pass
        # Forward pass (ensure float32 for Grad-CAM stability)
        # Note: We assume model is already in eval mode.
        # If the model was trained with AMP, it might have float16 weights if not fully cast back.
        # But usually model.load_state_dict loads fp32 unless saved as fp16. 
        # To be safe and avoid "Half Precision" errors:
        
        # This is a safe guard.
        # The calling code should ideally handle this, but we can enforce it here if needed.
        # But for now, let's just make sure we capture gradients.
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target = output[0, class_idx]
        target.backward()
        
        # Compute weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return cam.squeeze().cpu().numpy()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation.
    Improved version that uses weighted gradients.
    """
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target = output[0, class_idx]
        target.backward()
        
        # Grad-CAM++ weights
        grads = self.gradients
        acts = self.activations
        
        # Alpha calculation
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # Weights
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        
        # CAM
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return cam.squeeze().cpu().numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        image: Original image (H, W, 3) in range [0, 255]
        heatmap: CAM heatmap (H, W) in range [0, 1]
        alpha: Transparency for overlay
        colormap: OpenCV colormap
    
    Returns:
        Overlay image
    """
    # Convert heatmap to color
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Ensure image is in correct format
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay


def visualize_predictions(
    model,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_names: List[str],
    probabilities: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize predictions with Grad-CAM for each detected disease.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor (1, 3, H, W)
        original_image: Original image for display (H, W, 3)
        class_names: List of class names
        probabilities: Predicted probabilities per class
        threshold: Probability threshold for detection
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Get target layer for Grad-CAM
    target_layer = model.get_cam_target_layer()
    gradcam = GradCAMPlusPlus(model, target_layer)
    
    try:
        # Get positive predictions
        positive_indices = np.where(probabilities > threshold)[0]
        
        if len(positive_indices) == 0:
            positive_indices = [probabilities.argmax()]  # Show top prediction
        
        n_preds = len(positive_indices)
        n_cols = min(n_preds + 1, 4)
        n_rows = (n_preds + 1 + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Show original image
        ax = axes.flat[0]
        ax.imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
        ax.set_title('Original', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Show Grad-CAM for each positive prediction
        for i, class_idx in enumerate(positive_indices):
            ax = axes.flat[i + 1]
            
            # Generate CAM
            cam = gradcam.generate(image_tensor, class_idx)
            
            # Overlay
            overlay = overlay_heatmap(original_image, cam)
            ax.imshow(overlay)
            
            prob = probabilities[class_idx]
            name = class_names[class_idx]
            ax.set_title(f'{name}\n{prob:.2%}', fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # Hide empty axes
        for i in range(n_preds + 1, len(axes.flat)):
            axes.flat[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig

    finally:
        # CRITICAL: Remove hooks after usage to prevent memory leaks
        gradcam.remove_hooks()
        # Note: We don't close fig here because we return it, calling code should close it if needed,
        # or we accept that figures consume memory until closed. 
        # But hooks are the main leak source on the model.


def create_explanation_report(
    model,
    image_path: str,
    transform,
    class_names: List[str],
    device: torch.device,
    output_dir: str
) -> dict:
    """
    Create full explanation report for a single image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        transform: Preprocessing transform
        class_names: List of class names
        device: Torch device
        output_dir: Output directory for report
    
    Returns:
        Dictionary with predictions and paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    original_image = np.array(Image.open(image_path).convert('RGB'))
    image_tensor = transform(Image.open(image_path).convert('L').convert('RGB'))
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    # Generate visualization
    image_name = Path(image_path).stem
    save_path = os.path.join(output_dir, f'{image_name}_gradcam.png')
    
    visualize_predictions(
        model=model,
        image_tensor=image_tensor,
        original_image=original_image,
        class_names=class_names,
        probabilities=probabilities,
        threshold=0.3,
        save_path=save_path
    )
    
    # Return report
    report = {
        'image_path': image_path,
        'predictions': {
            name: float(prob)
            for name, prob in zip(class_names, probabilities)
        },
        'gradcam_path': save_path
    }
    
    return report


if __name__ == "__main__":
    print("Grad-CAM visualization module loaded.")
    print("Use visualize_predictions() or create_explanation_report() for explainability.")
