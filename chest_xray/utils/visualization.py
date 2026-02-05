"""
Visualization Utilities for Error Analysis
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import os

# Import GradCAM from the existing module
from .gradcam import GradCAM, GradCAMPlusPlus 

def show_cam_on_image(img_numpy: np.ndarray, mask: np.ndarray, use_rgb: bool = True, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on image.
    args:
        img_numpy: (H, W, 3) float [0, 1]
        mask: (H, W) float [0, 1]
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    heatmap = np.float32(heatmap) / 255
    
    if np.max(img_numpy) > 1:
        raise ValueError("img_numpy should be float in [0, 1]")
        
    cam = heatmap + img_numpy
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def save_error_grid(images, cams, captions, save_path, max_cols=5):
    """
    Save a grid of images with Grad-CAM overlays.
    
    Args:
        images: List of numpy images (H, W, 3) in [0, 1]
        cams: List of CAM masks (H, W) in [0, 1]
        captions: List of strings
        save_path: Output path
    """
    count = len(images)
    if count == 0:
        return

    cols = min(count, max_cols)
    rows = (count + cols - 1) // cols
    
    # Increase height per row to accommodate captions
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows + 1))
    
    # Handle single image case
    if count == 1:
        axes = np.array([axes])
    
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
    for i, ax in enumerate(axes):
        if i < count:
            img = images[i]
            cam = cams[i]
            cap = captions[i]
            
            # Prepare Heatmap & Overlay
            # Note: We implement the overlay logic here directly as requested by user pattern
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = np.float32(heatmap) / 255
            
            # Overlay (0.6 original + 0.4 heatmap)
            overlay = 0.6 * img + 0.4 * heatmap
            overlay = np.clip(overlay, 0, 1)
            
            ax.imshow(overlay)
            ax.set_title(cap, fontsize=9, wrap=True)
            ax.axis('off')
        else:
            ax.axis('off') # Hide empty subplots
            
    plt.tight_layout()
    # mkdir if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_metrics_curves(targets, preds, class_names, output_dir):
    """
    Plot ROC and Precision-Recall curves for each class.
    
    Args:
        targets: (N, C) binary targets
        preds: (N, C) probabilities
        class_names: List of class names
        output_dir: Output directory
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    # Create directory
    curve_dir = os.path.join(output_dir, 'curves')
    os.makedirs(curve_dir, exist_ok=True)
    
    # 1. ROC Curves
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(targets[:, i], preds[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        except:
            pass # Skip if class not present
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(curve_dir, 'roc_curves.png'), dpi=150)
    plt.close()
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(class_names):
        try:
            precision, recall, _ = precision_recall_curve(targets[:, i], preds[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')
        except:
            pass
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left", fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(curve_dir, 'pr_curves.png'), dpi=150)
    plt.close()
