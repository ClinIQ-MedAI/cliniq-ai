from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml

# --- CONFIGURATION ---
DATA_YAML = "/N/u/moabouag/Quartz/Documents/cliniq/wrest/train/detection/data.yaml"
PROJECT_DIR = "/N/scratch/moabouag/grazpedwri/runs/train"
MODEL_NAME = "yolov8n_graz_640_optimized"

# Load class names from data.yaml
with open(DATA_YAML, 'r') as f:
    data_config = yaml.safe_load(f)
CLASS_NAMES = data_config['names']


def on_fit_epoch_end(trainer):
    """Custom callback to display detailed per-class metrics after each validation epoch."""
    metrics = trainer.metrics
    epoch = trainer.epoch + 1
    
    if not hasattr(trainer, 'validator') or trainer.validator is None:
        return
    
    # Get per-class metrics from validator results
    validator = trainer.validator
    
    # Check if we have per-class data
    if hasattr(validator, 'metrics') and hasattr(validator.metrics, 'results_dict'):
        results = validator.metrics.results_dict
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} - DETAILED PER-CLASS METRICS")
        print(f"{'='*80}")
        
        # Print header
        header = f"{'Class':<20} {'AP50':>10} {'AP50-95':>10} {'Precision':>10} {'Recall':>10}"
        print(header)
        print("-" * 60)
        
        # Get per-class AP values if available
        if hasattr(validator.metrics, 'ap50') and hasattr(validator.metrics, 'ap'):
            ap50_per_class = validator.metrics.ap50
            ap_per_class = validator.metrics.ap  # AP50-95
            
            # Get precision and recall per class
            p_per_class = validator.metrics.p if hasattr(validator.metrics, 'p') else None
            r_per_class = validator.metrics.r if hasattr(validator.metrics, 'r') else None
            
            for i, class_name in CLASS_NAMES.items():
                if i < len(ap50_per_class):
                    ap50 = ap50_per_class[i] if ap50_per_class[i] is not None else 0.0
                    ap = ap_per_class[i] if ap_per_class[i] is not None else 0.0
                    
                    # Handle precision and recall
                    prec = p_per_class[i] if (p_per_class is not None and i < len(p_per_class)) else 0.0
                    rec = r_per_class[i] if (r_per_class is not None and i < len(r_per_class)) else 0.0
                    
                    # Color coding based on AP50 performance
                    if ap50 >= 0.7:
                        status = "✅"
                    elif ap50 >= 0.4:
                        status = "🟡"
                    else:
                        status = "🔴"
                    
                    print(f"{status} {class_name:<17} {ap50:>10.4f} {ap:>10.4f} {prec:>10.4f} {rec:>10.4f}")
            
            # Print summary statistics
            print("-" * 60)
            mean_ap50 = np.mean([ap50_per_class[i] for i in range(len(CLASS_NAMES)) if i < len(ap50_per_class)])
            mean_ap = np.mean([ap_per_class[i] for i in range(len(CLASS_NAMES)) if i < len(ap_per_class)])
            print(f"{'MEAN':<20} {mean_ap50:>10.4f} {mean_ap:>10.4f}")
        
        print(f"{'='*80}\n")


def main():
    print(f"--- Starting Training: {MODEL_NAME} ---")
    print("🚀 Using YOLOv8n (nano) - Optimized for speed and GPU efficiency")
    
    # Start fresh with YOLOv8n (nano) model
    print("📦 Loading YOLOv8n pre-trained model...")
    model = YOLO('yolov8n.pt')
    
    # Register custom callback for per-class metrics
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    print("✅ Registered per-class metrics callback")
    
    # Check device
    device = 0 if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️  WARNING: CUDA not available. Training will be SLOW.")
    
    # Train with optimized settings for YOLOv8n
    # YOLOv8n is ~25x smaller than YOLOv8x, allowing larger batch sizes
    results = model.train(
        data=DATA_YAML,
        epochs=150,             # More epochs for nano model to converge
        imgsz=640,              # Standard YOLO size - optimal for nano model
        batch=16,               # Larger batch size (nano model is very small)
        workers=4,              # Parallel data loading for speed
        project=PROJECT_DIR,
        name=MODEL_NAME,
        exist_ok=True,
        
        # Early Stopping & Checkpointing
        patience=20,            # Early stopping patience
        save=True,
        save_period=10,         # Save checkpoint every 10 epochs
        
        # Learning Rate Schedule (optimized for nano)
        lr0=0.01,               # Standard learning rate for nano
        lrf=0.01,               # Final LR = lr0 * lrf (cosine decay)
        warmup_epochs=3,        # Warmup to stabilize training
        
        # Optimizer
        optimizer='SGD',        # SGD is recommended for YOLO training
        momentum=0.937,
        weight_decay=0.0005,
        
        # X-ray Augmentations (conservative)
        augment=True,
        mosaic=1.0,             # Full mosaic augmentation
        mixup=0.0,              # Disable mixup for medical images
        close_mosaic=15,        # Disable mosaic in last 15 epochs
        fliplr=0.5,             # Horizontal flip OK
        flipud=0.0,             # No vertical flip for X-ray
        degrees=5.0,            # Slight rotation
        translate=0.1,          # Slight translation
        scale=0.3,              # Scale jitter
        hsv_v=0.2,              # Brightness variation
        hsv_h=0.0,              # No hue shift (grayscale X-ray)
        hsv_s=0.0,              # No saturation shift
        
        # Performance Optimization
        device=device,
        verbose=True,
        amp=True,               # Enable AMP for faster training
        cache='ram',            # Cache images in RAM for speed
        
        # Visualization
        plots=True              # Enable built-in training plots
    )
    
    print("Training Complete.")
    print(f"Results saved to {PROJECT_DIR}/{MODEL_NAME}")
    
    # Generate custom training curves
    plot_training_curves(os.path.join(PROJECT_DIR, MODEL_NAME))


def plot_training_curves(run_dir):
    """Generate detailed training visualization after training completes."""
    results_csv = os.path.join(run_dir, "results.csv")
    
    if not os.path.exists(results_csv):
        print("No results.csv found, skipping custom plots.")
        return
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # Clean column names
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress', fontsize=14)
    
    # Box Loss
    if 'train/box_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train')
        if 'val/box_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Class Loss
    if 'train/cls_loss' in df.columns:
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train')
        if 'val/cls_loss' in df.columns:
            axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # DFL Loss
    if 'train/dfl_loss' in df.columns:
        axes[0, 2].plot(df['epoch'], df['train/dfl_loss'], label='Train')
        if 'val/dfl_loss' in df.columns:
            axes[0, 2].plot(df['epoch'], df['val/dfl_loss'], label='Val')
        axes[0, 2].set_title('DFL Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # mAP50
    if 'metrics/mAP50(B)' in df.columns:
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], color='green')
        axes[1, 0].set_title('mAP@50')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
    
    # mAP50-95
    if 'metrics/mAP50-95(B)' in df.columns:
        axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], color='blue')
        axes[1, 1].set_title('mAP@50-95')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
    
    # Precision & Recall
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        axes[1, 2].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        axes[1, 2].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        axes[1, 2].set_title('Precision & Recall')
        axes[1, 2].legend()
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curves_custom.png"), dpi=150)
    plt.close()
    print(f"Custom training curves saved to {run_dir}/training_curves_custom.png")


if __name__ == "__main__":
    main()
