#!/usr/bin/env python3
"""
Error Analysis Script for Chest X-ray Classification
- Loads trained model
- runs inference on Validation Set (with metadata)
- Computes detailed metrics
- Generates Grad-CAM for top False Positives and False Negatives
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from data.dataset import get_dataloaders, ChestXrayDataset
from data.transforms import get_val_transforms
from models.convnext import ConvNeXtClassifier
from utils.visualization import GradCAM, save_error_grid, plot_metrics_curves

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'error_analysis.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_data_with_metadata(cfg):
    """
    Manually create validation loader with metadata enabled.
    """
    # Load raw dataframe to get splits
    # We want to use the same split logic as training
    # Easier way: reuse get_dataloaders logic but override dataset parameters
    
    # We'll rely on get_dataloaders to do the splitting, then access the index/lists
    # This is a bit duplicative but safest to ensure identical splits
    _, val_loader_orig, test_loader, train_dataset = get_dataloaders(
        batch_size=cfg.training.batch_size,
        image_size=cfg.training.image_size,
        num_workers=cfg.hardware.num_workers,
        use_official_split=True,  # Assuming this matches train.py
        seed=cfg.hardware.seed
    )
    
    # Extract validation images list from the original validation loader's dataset
    val_images = val_loader_orig.dataset.image_list
    labels_df = val_loader_orig.dataset.labels_df
    
    # specific transform
    val_transform = get_val_transforms(cfg.training.image_size)
    
    # Create new validation dataset with return_metadata=True
    val_dataset_meta = ChestXrayDataset(
        root_dir=cfg.data.root_dir,
        image_list=val_images,
        labels_df=labels_df,
        classes=cfg.data.classes,
        transform=val_transform,
        return_metadata=True
    )
    
    val_loader_meta = torch.utils.data.DataLoader(
        val_dataset_meta,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.hardware.num_workers,
        pin_memory=True
    )
    
    return val_loader_meta, cfg.data.classes

def inverse_normalize(tensor):
    """Convert normalized tensor (B, C, H, W) to numpy (B, H, W, 3) in [0,1]."""
    # ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img

@torch.no_grad()
def run_inference(model, loader, device, debug=False):
    model.eval()
    
    all_preds = []
    all_targets = []
    all_paths = []
    
    print("Running inference...")
    for i, batch in enumerate(tqdm(loader)):
        images, targets, paths = batch
        images = images.to(device)
        
        logits = model(images)
        probs = torch.sigmoid(logits)
        
        all_preds.append(probs.cpu().numpy())
        all_targets.append(targets.numpy())
        all_paths.extend(paths)
        
        if debug and i >= 5:
            print("Debug mode: stopping after 5 batches")
            break
        
    return np.concatenate(all_preds), np.concatenate(all_targets), all_paths

def analyze_errors(
    model, 
    preds, 
    targets, 
    paths, 
    class_names, 
    output_dir, 
    device,
    thresholds=None
):
    logger = logging.getLogger(__name__)
    
    if thresholds is None:
        thresholds = {c: 0.5 for c in class_names}
    
    # 1. Global Metrics
    try:
        mean_auc = roc_auc_score(targets, preds, average='macro')
        logger.info(f"Mean AUC: {mean_auc:.4f}")
    except:
        logger.warning("Could not calculate AUC (single class present?)")
        
    # Plot Curves
    try:
        plot_metrics_curves(targets, preds, class_names, output_dir)
        logger.info("Saved metric curves to curves/")
    except Exception as e:
        logger.warning(f"Could not plot curves: {e}")
        
    # 2. Per-Class Analysis
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    report_lines = []
    report_lines.append("# Error Analysis Report")
    report_lines.append(f"Output Directory: {output_dir}")
    
    # Setup Grad-CAM
    target_layer = model.get_cam_target_layer()
    grad_cam = GradCAM(model, target_layer)
    
    # Prepare transforms for visualization (reload images)
    # We just read them with opencv/PIL when needed
    
    for i, class_name in enumerate(class_names):
        cls_preds = preds[:, i]
        cls_targets = targets[:, i]
        cls_thresh = thresholds.get(class_name, 0.5)
        
        cls_bin_preds = (cls_preds >= cls_thresh).astype(int)
        
        # Metrics
        try:
            auc = roc_auc_score(cls_targets, cls_preds)
        except:
            auc = 0.0
        acc = accuracy_score(cls_targets, cls_bin_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(cls_targets, cls_bin_preds, average='binary', zero_division=0)
        
        log_str = f"Class: {class_name:20s} | AUC: {auc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Thresh: {cls_thresh:.3f}"
        logger.info(log_str)
        report_lines.append(f"\n## {class_name}")
        report_lines.append(f"- AUC: {auc:.4f}")
        report_lines.append(f"- F1: {f1:.4f} (at threshold {cls_thresh:.3f})")
        report_lines.append(f"- Precision: {prec:.4f}")
        report_lines.append(f"- Recall: {rec:.4f}")
        report_lines.append(f"- Counts: Pos={int(cls_targets.sum())}, Neg={len(cls_targets)-int(cls_targets.sum())}")
        
        # Identify FP and FN
        # FP: Predicted=1 (and high prob), Truth=0
        fp_mask = (cls_targets == 0) & (cls_bin_preds == 1)
        fp_indices = np.where(fp_mask)[0]
        # Sort by confidence (descending) - confident wrong prediction
        fp_indices = fp_indices[np.argsort(cls_preds[fp_indices])[::-1]]
        
        # FN: Predicted=0 (and low prob), Truth=1
        fn_mask = (cls_targets == 1) & (cls_bin_preds == 0)
        fn_indices = np.where(fn_mask)[0]
        # Sort by confidence (ascending) - confident wrong prediction (lowest prob for positive class)
        fn_indices = fn_indices[np.argsort(cls_preds[fn_indices])]
        
        # Visualize Top 5 FP
        if len(fp_indices) > 0:
            top_fps = fp_indices[:5]
            fp_images = []
            fp_cams = []
            fp_captions = []
            
            for idx in top_fps:
                img_path = paths[idx]
                prob = cls_preds[idx]
                
                # Load and preprocess
                img_raw = cv2.imread(img_path)
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                img_raw = cv2.resize(img_raw, (512, 512)) # Match model input
                img_float = img_raw.astype(np.float32) / 255.0
                
                # Create tensor for CAM
                # Normalize using same stats as training
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
                img_norm = (img_float - mean) / std
                img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
                # Generate CAM
                cam = grad_cam.generate(img_tensor, class_idx=i)
                
                fp_images.append(img_float)
                fp_cams.append(cam)
                fp_captions.append(f"FP {class_name}\nProb: {prob:.3f}\nFile: {os.path.basename(img_path)}")
                
            save_path = os.path.join(output_dir, 'visualizations', f'{class_name}_FP.png')
            save_error_grid(fp_images, fp_cams, fp_captions, save_path)
            report_lines.append(f"- [Top 5 False Positives](./visualizations/{class_name}_FP.png)")
        else:
            report_lines.append("- No False Positives found.")

        # Visualize Top 5 FN
        if len(fn_indices) > 0:
            top_fns = fn_indices[:5]
            fn_images = []
            fn_cams = []
            fn_captions = []
            
            for idx in top_fns:
                img_path = paths[idx]
                prob = cls_preds[idx]
                
                # Load and preprocess
                img_raw = cv2.imread(img_path)
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                img_raw = cv2.resize(img_raw, (512, 512))
                img_float = img_raw.astype(np.float32) / 255.0
                
                # Normalize
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
                img_norm = (img_float - mean) / std
                img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
                # Generate CAM
                cam = grad_cam.generate(img_tensor, class_idx=i)
                
                fn_images.append(img_float)
                fn_cams.append(cam)
                fn_captions.append(f"FN {class_name}\nProb: {prob:.3f}\nFile: {os.path.basename(img_path)}")
                
            save_path = os.path.join(output_dir, 'visualizations', f'{class_name}_FN.png')
            save_error_grid(fn_images, fn_cams, fn_captions, save_path)
            report_lines.append(f"- [Top 5 False Negatives](./visualizations/{class_name}_FN.png)")
        else:
            report_lines.append("- No False Negatives found.")
            
    # Save Report
    with open(os.path.join(output_dir, 'report.md'), 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Analysis saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Error Analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='error_analysis_results', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Run on small subset for debugging')
    args = parser.parse_args()
    
    # Config
    cfg = config
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.output)
    
    # Load Data
    logger.info("Loading validation data...")
    val_loader, class_names = load_data_with_metadata(cfg)
    
    # Load Model
    logger.info(f"Loading model from {args.checkpoint}...")
    model = ConvNeXtClassifier(
        num_classes=len(class_names),
        pretrained=False, # Weights loaded from checkpoint
        dropout=cfg.model.dropout
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get optimal thresholds if available
    thresholds = checkpoint.get('thresholds', None)
    if thresholds:
        logger.info(f"Loaded optimal thresholds: {json.dumps(thresholds, indent=2)}")
    else:
        logger.info("No thresholds found in checkpoint, using 0.5")
    
    # Run Inference
    preds, targets, paths = run_inference(model, val_loader, device, debug=args.debug)
    
    # Run Analysis
    analyze_errors(
        model, preds, targets, paths, 
        class_names, args.output, device, 
        thresholds=thresholds
    )

if __name__ == '__main__':
    main()
