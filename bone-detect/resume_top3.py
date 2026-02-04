"""
Resume YOLO TOP-3 Training from last checkpoint.

Usage:
    python resume_top3.py
"""

import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from ultralytics import YOLO


# Latest checkpoint
CHECKPOINT = Path(__file__).parent / "outputs/YOLO11x_TOP3_20260203_0645/weights/last.pt"
DATA_YAML = Path(__file__).parent / "data_top3.yaml"

# Training params
EPOCHS = 100  # Total epochs target
IMG_SIZE = 800
BATCH_SIZE = 24

CLASS_NAMES = ["fracture", "metal", "periostealreaction"]

# Same output dir to continue
OUTPUT_DIR = Path(__file__).parent / "outputs" / "YOLO11x_TOP3_20260203_0645"

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
        
        mAP50 = float(metrics.box.map50)
        mAP50_95 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        
        history["epoch"].append(epoch)
        history["mAP50"].append(mAP50)
        history["mAP50_95"].append(mAP50_95)
        history["precision"].append(precision)
        history["recall"].append(recall)
        
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
        
        # Save plots
        save_training_plots()
        
        with open(OUTPUT_DIR / "metrics_history_resumed.json", "w") as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        print(f"Callback error: {e}")


def save_training_plots():
    """Save training progress plots."""
    if len(history["epoch"]) < 1:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = history["epoch"]
    
    axes[0, 0].plot(epochs, history["mAP50"], 'b-', label="mAP@0.5", linewidth=2)
    axes[0, 0].plot(epochs, history["mAP50_95"], 'r-', label="mAP@0.5:0.95", linewidth=2)
    axes[0, 0].set_title("mAP Scores (Resumed)", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("mAP")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    axes[0, 1].plot(epochs, history["precision"], 'g-', label="Precision", linewidth=2)
    axes[0, 1].plot(epochs, history["recall"], 'm-', label="Recall", linewidth=2)
    axes[0, 1].set_title("Precision & Recall", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
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
    plt.savefig(OUTPUT_DIR / "training_progress_resumed.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("█"*70)
    print("RESUMING YOLOv11x TOP-3 TRAINING")
    print("█"*70)
    print(f"\nCheckpoint: {CHECKPOINT}")
    print(f"Data: {DATA_YAML}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Load from checkpoint
    model = YOLO(str(CHECKPOINT))
    
    # Add callback
    model.add_callback("on_val_end", on_val_end)
    
    # Resume training
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        resume=True,  # Resume from checkpoint
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        workers=4,
        device=0,
    )
    
    print("\n" + "█"*70)
    print("🎉 TRAINING RESUMED AND COMPLETE! 🎉")
    print("█"*70)
    
    if history["mAP50"]:
        best_map50 = max(history["mAP50"])
        print(f"\nBest mAP@0.5 in resumed run: {best_map50:.4f}")
        
        print(f"\nFinal Per-Class AP@0.5:")
        for name in CLASS_NAMES:
            if history["class_ap50"][name]:
                ap = history["class_ap50"][name][-1]
                bar = "█" * int(ap * 20)
                print(f"  {name:20s}: {ap:.4f} {bar}")
    
    print(f"\nModel saved to: {OUTPUT_DIR / 'weights/best.pt'}")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()
