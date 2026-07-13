"""
Evaluation Metrics for Multi-Label Classification
- AUC-ROC per class
- PR-AUC for rare diseases
- Threshold optimization
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve
)
from typing import Dict, List, Optional, Tuple


def compute_auc_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute AUC-ROC for each class and mean.
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred: Predicted probabilities (N, num_classes)
        class_names: Optional list of class names
    
    Returns:
        Dictionary with per-class and mean AUC-ROC
    """
    num_classes = y_true.shape[1]
    class_names = class_names or [f"class_{i}" for i in range(num_classes)]
    
    results = {}
    valid_aucs = []
    
    for i, name in enumerate(class_names):
        # Skip if no positive samples
        if y_true[:, i].sum() == 0:
            results[name] = float('nan')
            continue
        
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            results[name] = auc
            valid_aucs.append(auc)
        except ValueError:
            results[name] = float('nan')
    
    # Mean AUC
    results['mean_auc'] = np.mean(valid_aucs) if valid_aucs else float('nan')
    
    return results


def compute_pr_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute PR-AUC (Average Precision) for each class.
    PR-AUC is better for evaluating rare diseases.
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred: Predicted probabilities (N, num_classes)
        class_names: Optional list of class names
    
    Returns:
        Dictionary with per-class and mean PR-AUC
    """
    num_classes = y_true.shape[1]
    class_names = class_names or [f"class_{i}" for i in range(num_classes)]
    
    results = {}
    valid_aps = []
    
    for i, name in enumerate(class_names):
        if y_true[:, i].sum() == 0:
            results[f"{name}_pr_auc"] = float('nan')
            continue
        
        try:
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            results[f"{name}_pr_auc"] = ap
            valid_aps.append(ap)
        except ValueError:
            results[f"{name}_pr_auc"] = float('nan')
    
    results['mean_pr_auc'] = np.mean(valid_aps) if valid_aps else float('nan')
    
    return results


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    method: str = 'f1'
) -> Dict[str, float]:
    """
    Find optimal threshold for each class based on F1-score.
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred: Predicted probabilities (N, num_classes)
        class_names: Optional list of class names
        method: 'f1' or 'youden' (Youden's J statistic)
    
    Returns:
        Dictionary with optimal thresholds per class
    """
    num_classes = y_true.shape[1]
    class_names = class_names or [f"class_{i}" for i in range(num_classes)]
    
    thresholds = {}
    
    for i, name in enumerate(class_names):
        if y_true[:, i].sum() == 0:
            thresholds[name] = 0.5
            continue
        
        if method == 'youden':
            # Use ROC curve to find optimal threshold (Youden's J)
            fpr, tpr, thresh = roc_curve(y_true[:, i], y_pred[:, i])
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            thresholds[name] = thresh[best_idx]
        else:
            # Grid search for best F1
            best_f1 = 0
            best_thresh = 0.5
            # Finer search range for rare diseases
            for t in np.arange(0.01, 0.9, 0.01):
                preds_binary = (y_pred[:, i] > t).astype(int)
                if preds_binary.sum() == 0:
                    continue
                f1 = f1_score(y_true[:, i], preds_binary, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t
            thresholds[name] = best_thresh
    
    return thresholds


def compute_metrics_with_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Dict[str, float],
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute F1, precision, recall using per-class thresholds.
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred: Predicted probabilities (N, num_classes)
        thresholds: Per-class thresholds
        class_names: List of class names
    
    Returns:
        Dictionary with metrics
    """
    num_classes = y_true.shape[1]
    class_names = class_names or [f"class_{i}" for i in range(num_classes)]
    
    # Apply per-class thresholds
    y_pred_binary = np.zeros_like(y_pred)
    for i, name in enumerate(class_names):
        thresh = thresholds.get(name, 0.5)
        y_pred_binary[:, i] = (y_pred[:, i] > thresh).astype(int)
    
    results = {}
    
    # Per-class metrics
    f1s, precs, recs = [], [], []
    for i, name in enumerate(class_names):
        if y_true[:, i].sum() == 0:
            continue
        
        f1 = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        prec = precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        
        results[f"{name}_f1"] = f1
        results[f"{name}_precision"] = prec
        results[f"{name}_recall"] = rec
        
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
    
    # Macro averages
    results['macro_f1'] = np.mean(f1s) if f1s else 0
    results['macro_precision'] = np.mean(precs) if precs else 0
    results['macro_recall'] = np.mean(recs) if recs else 0
    
    return results


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all standard metrics.
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred: Predicted probabilities (N, num_classes)
        class_names: List of class names
        threshold: Fixed threshold (for quick eval)
    
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    # AUC-ROC
    auc_results = compute_auc_roc(y_true, y_pred, class_names)
    results.update(auc_results)
    
    # PR-AUC
    pr_results = compute_pr_auc(y_true, y_pred, class_names)
    results.update(pr_results)
    
    # F1 with fixed threshold
    y_pred_binary = (y_pred > threshold).astype(int)
    results['f1_macro'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    results['f1_micro'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    return results


class MetricTracker:
    """Track metrics during training."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        self.all_preds = []
        self.all_targets = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Add batch of predictions and targets."""
        self.all_preds.append(preds.detach().cpu().numpy())
        self.all_targets.append(targets.detach().cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        y_pred = np.concatenate(self.all_preds, axis=0)
        y_true = np.concatenate(self.all_targets, axis=0)
        
        # Apply sigmoid if logits
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = 1 / (1 + np.exp(-y_pred))
        
        return compute_all_metrics(y_true, y_pred, self.class_names)


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    num_samples = 100
    num_classes = 13
    
    y_true = np.random.randint(0, 2, (num_samples, num_classes))
    y_pred = np.random.rand(num_samples, num_classes)
    
    class_names = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Infiltration",
        "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
    ]
    
    # Compute AUC-ROC
    auc_results = compute_auc_roc(y_true, y_pred, class_names)
    print("AUC-ROC per class:")
    for name in class_names:
        print(f"  {name}: {auc_results[name]:.4f}")
    print(f"  Mean AUC: {auc_results['mean_auc']:.4f}")
    
    # Find optimal thresholds
    thresholds = find_optimal_thresholds(y_true, y_pred, class_names)
    print("\nOptimal thresholds:")
    for name, thresh in thresholds.items():
        print(f"  {name}: {thresh:.3f}")
