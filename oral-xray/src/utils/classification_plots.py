"""
Classification plots and reports for dental X-ray crop classification.
Similar structure to yolo_plots.py for consistency.
"""

import json
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_training_curves(history: Dict, out_dir: Path):
    """
    Plot training and validation curves.
    
    Args:
        history: Dict with train_loss, val_loss, train_acc, val_acc
        out_dir: Output directory
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    
    if not epochs:
        return
    
    # Loss curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    if "train_loss" in history:
        plt.plot(epochs, history["train_loss"], 'b-', label='Train Loss', linewidth=2)
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], 'r-', label='Val Loss', linewidth=2)
    plt.title('Training & Validation Loss', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    if "train_acc" in history:
        plt.plot(epochs, history["train_acc"], 'b-', label='Train Acc', linewidth=2)
    if "val_acc" in history:
        plt.plot(epochs, history["val_acc"], 'r-', label='Val Acc', linewidth=2)
    plt.title('Training & Validation Accuracy', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=240)
    plt.close()
    
    # Separate plots for presentation
    plt.figure(figsize=(8, 6))
    if "train_loss" in history:
        plt.plot(epochs, history["train_loss"], 'b-', label='Train', linewidth=2)
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], 'r-', label='Validation', linewidth=2)
    plt.title('Loss Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=240)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    if "train_acc" in history:
        plt.plot(epochs, history["train_acc"], 'b-', label='Train', linewidth=2)
    if "val_acc" in history:
        plt.plot(epochs, history["val_acc"], 'r-', label='Validation', linewidth=2)
    plt.title('Accuracy Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=240)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray, 
    class_names: List[str], 
    out_path: Path,
    normalize: bool = True,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array (num_classes x num_classes)
        class_names: List of class names
        out_path: Output file path
        normalize: Whether to normalize the matrix
        title: Plot title
    """
    if normalize:
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        cm_display = cm_norm
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if HAS_SEABORN:
        sns.heatmap(
            cm_display, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
    else:
        im = ax.imshow(cm_display, cmap='Blues')
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_display[i, j]
                text = f'{val:{fmt}}'
                ax.text(j, i, text, ha='center', va='center', fontsize=8)
        
        plt.colorbar(im)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close()


def plot_per_class_metrics(
    metrics: Dict[str, Dict[str, float]], 
    out_dir: Path
):
    """
    Plot per-class precision, recall, F1 scores.
    
    Args:
        metrics: Dict mapping class_name -> {precision, recall, f1, support}
        out_dir: Output directory
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = list(metrics.keys())
    precision = [metrics[c].get('precision', 0) for c in class_names]
    recall = [metrics[c].get('recall', 0) for c in class_names]
    f1 = [metrics[c].get('f1', 0) for c in class_names]
    support = [metrics[c].get('support', 0) for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    # Precision, Recall, F1 grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_metrics.png", dpi=240)
    plt.close()
    
    # Support (sample count) bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(class_names)))
    ax.bar(x, support, color=colors)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('Validation Samples per Class', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(support):
        ax.text(i, v + max(support) * 0.02, str(v), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution_val.png", dpi=240)
    plt.close()


def plot_class_distribution(
    train_counts: Dict[str, int],
    val_counts: Dict[str, int],
    out_dir: Path
):
    """
    Plot class distribution for train and val sets.
    
    Args:
        train_counts: Dict mapping class_name -> count for train
        val_counts: Dict mapping class_name -> count for val
        out_dir: Output directory
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = list(train_counts.keys())
    train_vals = [train_counts.get(c, 0) for c in class_names]
    val_vals = [val_counts.get(c, 0) for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_vals, width, label='Train', color='#3498db')
    bars2 = ax.bar(x + width/2, val_vals, width, label='Validation', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('Class Distribution (Train vs Validation)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=240)
    plt.close()


def save_sample_predictions(
    images: List[Path],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    out_dir: Path,
    samples_per_class: int = 4
):
    """
    Save sample predictions with true/predicted labels.
    
    Args:
        images: List of image paths
        true_labels: List of true class names
        pred_labels: List of predicted class names
        confidences: List of prediction confidences
        out_dir: Output directory
        samples_per_class: Number of samples per class to show
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by true class
    by_class = {}
    for img, true, pred, conf in zip(images, true_labels, pred_labels, confidences):
        if true not in by_class:
            by_class[true] = []
        by_class[true].append((img, pred, conf))
    
    # For each class, save correct and incorrect samples
    for cls_name, samples in by_class.items():
        cls_dir = out_dir / cls_name.replace(" ", "_")
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        correct = [(img, pred, conf) for img, pred, conf in samples if pred == cls_name]
        incorrect = [(img, pred, conf) for img, pred, conf in samples if pred != cls_name]
        
        # Save some correct predictions
        for i, (img_path, pred, conf) in enumerate(correct[:samples_per_class]):
            try:
                img = Image.open(img_path)
                save_name = f"correct_{i:02d}_conf{conf:.2f}.jpg"
                img.save(cls_dir / save_name, quality=95)
            except Exception as e:
                pass
        
        # Save some incorrect predictions
        for i, (img_path, pred, conf) in enumerate(incorrect[:samples_per_class]):
            try:
                img = Image.open(img_path)
                save_name = f"wrong_{i:02d}_pred_{pred.replace(' ', '_')}_conf{conf:.2f}.jpg"
                img.save(cls_dir / save_name, quality=95)
            except Exception as e:
                pass


def generate_classification_report(
    history: Dict,
    confusion_matrix: np.ndarray,
    per_class_metrics: Dict[str, Dict[str, float]],
    class_names: List[str],
    train_counts: Dict[str, int],
    val_counts: Dict[str, int],
    config: Dict,
    out_dir: Path,
    best_epoch: int = None,
    best_acc: float = None
):
    """
    Generate full classification report with all plots and statistics.
    
    Args:
        history: Training history with loss and accuracy
        confusion_matrix: Confusion matrix array
        per_class_metrics: Dict of per-class metrics
        class_names: List of class names
        train_counts: Train set class counts
        val_counts: Val set class counts
        config: Training configuration
        out_dir: Output directory for all reports
        best_epoch: Best epoch number
        best_acc: Best validation accuracy
    """
    pres_dir = out_dir / "presentation"
    pres_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training curves
    plot_training_curves(history, pres_dir)
    
    # 2. Confusion matrix (normalized and raw)
    plot_confusion_matrix(
        confusion_matrix, class_names, 
        pres_dir / "confusion_matrix.png",
        normalize=True,
        title="Confusion Matrix (Normalized)"
    )
    plot_confusion_matrix(
        confusion_matrix, class_names,
        pres_dir / "confusion_matrix_raw.png", 
        normalize=False,
        title="Confusion Matrix (Counts)"
    )
    
    # 3. Per-class metrics
    plot_per_class_metrics(per_class_metrics, pres_dir)
    
    # 4. Class distribution
    plot_class_distribution(train_counts, val_counts, pres_dir / "dataset_report")
    
    # 5. Summary statistics
    overall_metrics = {
        "accuracy": best_acc,
        "best_epoch": best_epoch,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_samples": sum(train_counts.values()),
        "val_samples": sum(val_counts.values()),
    }
    
    # Calculate macro averages
    if per_class_metrics:
        overall_metrics["macro_precision"] = np.mean([m.get('precision', 0) for m in per_class_metrics.values()])
        overall_metrics["macro_recall"] = np.mean([m.get('recall', 0) for m in per_class_metrics.values()])
        overall_metrics["macro_f1"] = np.mean([m.get('f1', 0) for m in per_class_metrics.values()])
    
    (pres_dir / "summary.json").write_text(json.dumps(overall_metrics, indent=2))
    
    # 6. Per-class metrics as JSON
    (pres_dir / "per_class_metrics.json").write_text(json.dumps(per_class_metrics, indent=2))
    
    # 7. Config
    (pres_dir / "config.json").write_text(json.dumps(config, indent=2))
    
    print(f"✅ Classification report saved to: {pres_dir}")
    return pres_dir
