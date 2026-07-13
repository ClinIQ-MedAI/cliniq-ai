"""
YOLOv11x Training Script - TOP 3 CLASSES ONLY

Classes: fracture, metal, periostealreaction

Features:
- Per-class AP metrics printed after each epoch
- Live training graphs saved every epoch
- Final comprehensive report

Usage:
    python train_top3.py
"""

import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from ultralytics import YOLO


# Config
MODEL_NAME = "yolo11x.pt"
DATA_YAML = Path(__file__).parent / "data_top3.yaml"
EPOCHS = 100
IMG_SIZE = 1024  # Increased from 800 - helps detect subtle periosteal reaction
BATCH_SIZE = 16  # Reduced to accommodate higher resolution

CLASS_NAMES = ["fracture", "metal", "periostealreaction"]

# Output
OUTPUT_DIR = Path(__file__).parent / "outputs" / f"YOLO11x_TOP3_{datetime.now().strftime('%Y%m%d_%H%M')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# History for plotting
history = {
    "epoch": [],
    "mAP50": [],
    "mAP50_95": [],
    "precision": [],
    "recall": [],
    "class_ap50": {name: [] for name in CLASS_NAMES},
}


def on_val_end(validator):
    """Callback after validation - print per-class metrics."""
    try:
        metrics = validator.metrics
        epoch = len(history["epoch"]) + 1
        
        # Overall metrics
        mAP50 = float(metrics.box.map50)
        mAP50_95 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        
        history["epoch"].append(epoch)
        history["mAP50"].append(mAP50)
        history["mAP50_95"].append(mAP50_95)
        history["precision"].append(precision)
        history["recall"].append(recall)
        
        # Per-class AP@0.5
        print("\n" + "─"*60)
        print(f"EPOCH {epoch} - Per-Class AP@0.5:")
        print("─"*60)
        
        ap50_per_class = metrics.box.ap50
        for i, name in enumerate(CLASS_NAMES):
            if i < len(ap50_per_class):
                ap = float(ap50_per_class[i])
                history["class_ap50"][name].append(ap)
                bar = "█" * int(ap * 20)
                print(f"  {name:20s}: {ap:.4f} {bar}")
            else:
                history["class_ap50"][name].append(0.0)
        
        print(f"\n  mAP@0.5: {mAP50:.4f} | mAP@0.5:0.95: {mAP50_95:.4f}")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f}")
        print("─"*60)
        
        # Save live plots
        save_training_plots()
        
        # Save history to JSON
        with open(OUTPUT_DIR / "metrics_history.json", "w") as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        print(f"Callback error: {e}")


def save_training_plots():
    """Save training progress plots."""
    if len(history["epoch"]) < 1:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = history["epoch"]
    
    # mAP plots
    axes[0, 0].plot(epochs, history["mAP50"], 'b-', label="mAP@0.5", linewidth=2)
    axes[0, 0].plot(epochs, history["mAP50_95"], 'r-', label="mAP@0.5:0.95", linewidth=2)
    axes[0, 0].set_title("mAP Scores", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("mAP")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Precision/Recall
    axes[0, 1].plot(epochs, history["precision"], 'g-', label="Precision", linewidth=2)
    axes[0, 1].plot(epochs, history["recall"], 'm-', label="Recall", linewidth=2)
    axes[0, 1].set_title("Precision & Recall", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Per-class AP@0.5
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for i, name in enumerate(CLASS_NAMES):
        if history["class_ap50"][name]:
            axes[1, 0].plot(epochs, history["class_ap50"][name], 
                           color=colors[i], label=name, linewidth=2)
    axes[1, 0].set_title("Per-Class AP@0.5", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("AP@0.5")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Latest per-class bar chart
    if history["class_ap50"][CLASS_NAMES[0]]:
        latest_ap = [history["class_ap50"][name][-1] for name in CLASS_NAMES]
        bars = axes[1, 1].bar(CLASS_NAMES, latest_ap, color=colors, edgecolor='white', linewidth=2)
        axes[1, 1].set_title("Latest Per-Class AP@0.5", fontsize=14, fontweight="bold")
        axes[1, 1].set_ylabel("AP@0.5")
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        for bar, ap in zip(bars, latest_ap):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{ap:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_progress.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("█"*70)
    print("YOLOv11x TRAINING - TOP 3 CLASSES (with live metrics)")
    print("█"*70)
    print("\nClasses:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")
    print("="*70)
    
    print(f"\nData: {DATA_YAML}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}, Image: {IMG_SIZE}, Batch: {BATCH_SIZE}")
    
    # Load model
    model = YOLO(MODEL_NAME)
    
    # Add callback for per-class metrics
    model.add_callback("on_val_end", on_val_end)
    
    # Train with optimizations for weak classes
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        patience=25,  # Increased patience for convergence
        save=True,
        plots=True,
        verbose=True,
        workers=4,
        device=0,
        # Learning rate optimization
        lr0=0.005,      # Slightly lower initial LR for stability
        lrf=0.01,       # Final LR as fraction of lr0
        warmup_epochs=5,  # Longer warmup
        # Loss weights - boost classification for weak classes
        cls=2.0,        # Increase classification loss weight
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,      # Slightly increased mixup
        copy_paste=0.3,  # Add copy-paste augmentation for rare classes
        close_mosaic=15, # Keep mosaic active longer before disabling
    )
    
    # Final summary
    print("\n" + "█"*70)
    print("🎉 TRAINING COMPLETE! 🎉")
    print("█"*70)
    
    if history["mAP50"]:
        best_map50 = max(history["mAP50"])
        best_epoch = history["mAP50"].index(best_map50) + 1
        
        print(f"\nBest mAP@0.5: {best_map50:.4f} (epoch {best_epoch})")
        print(f"\nFinal Per-Class AP@0.5:")
        for name in CLASS_NAMES:
            if history["class_ap50"][name]:
                ap = history["class_ap50"][name][-1]
                bar = "█" * int(ap * 20)
                print(f"  {name:20s}: {ap:.4f} {bar}")
    
    print(f"\nOutputs:")
    print(f"  - Plots: {OUTPUT_DIR / 'training_progress.png'}")
    print(f"  - Metrics: {OUTPUT_DIR / 'metrics_history.json'}")
    print(f"  - Model: {OUTPUT_DIR / 'weights/best.pt'}")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()
