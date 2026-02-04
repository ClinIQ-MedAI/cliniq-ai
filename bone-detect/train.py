"""
YOLOv11x Training Script for Bone Fracture Detection

This script trains a YOLOv11x model on the GrazPedWri-DX dataset
with comprehensive visualizations and performance tracking.

Features:
- Live training progress plots (loss curves, mAP)
- Per-epoch visualization snapshots
- Test-Time Augmentation (TTA) for final evaluation
- Comprehensive JSON report with per-class metrics
- Final summary visualization

Usage:
    python train.py
"""

from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

# ==================== CONFIG ====================
DEVICE = 0  # GPU device (0 for first GPU, "cpu" for CPU)
MODEL_NAME = "yolo11x.pt"  # Model weights (yolo11x or yolov8x)

# Dataset path
DATA_YAML = Path(__file__).parent / "dataset" / "data.yaml"
if not DATA_YAML.exists():
    DATA_YAML = Path(__file__).parent / "data.yaml"

MODEL_NAME = "yolo11x.pt" 
EPOCHS = 200
IMGSZ = 1280 
BATCH = 20
PATIENCE = 25  

# Output directories
RUN_NAME = f"YOLOv11x_BONE_SOTA_{datetime.now().strftime('%Y%m%d_%H%M')}"
PROJECT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR = PROJECT_DIR / RUN_NAME
PLOTS_DIR = OUTPUT_DIR / "progress_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Class names for visualization
CLASS_NAMES = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture",
    "metal", "periostealreaction", "pronatorsign", "softtissue"
]

# Colors for visualization (medical-friendly palette)
CLASS_COLORS = [
    '#e74c3c',  # boneanomaly - red
    '#9b59b6',  # bonelesion - purple
    '#3498db',  # foreignbody - blue
    '#27ae60',  # fracture - green (main class)
    '#f39c12',  # metal - orange
    '#1abc9c',  # periostealreaction - teal
    '#e91e63',  # pronatorsign - pink
    '#00bcd4',  # softtissue - cyan
]

# ==================== HISTORY STORAGE ====================
history = {
    "epoch": [],
    "mAP50": [],
    "mAP50_95": [],
    "precision": [],
    "recall": [],
    "box_loss": [],
    "cls_loss": [],
    "dfl_loss": [],
}


# ==================== VISUALIZATION FUNCTIONS ====================
def create_progress_plot(history: dict, output_path: Path, epoch: int):
    """Create comprehensive training progress visualization."""
    
    if len(history["epoch"]) < 2:
        return
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid for plots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. mAP Progress (main metric)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history["epoch"], history["mAP50"], 'g-', linewidth=2.5, 
             marker='o', markersize=3, label="mAP@0.5", alpha=0.9)
    ax1.plot(history["epoch"], history["mAP50_95"], 'purple', linewidth=2.5,
             marker='s', markersize=3, label="mAP@0.5:0.95", alpha=0.9)
    ax1.fill_between(history["epoch"], 0, history["mAP50"], alpha=0.1, color='green')
    ax1.set_title("Validation mAP Progress", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("mAP", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(1, max(history["epoch"]))
    
    # Add current best annotation
    best_idx = np.argmax(history["mAP50"])
    best_epoch = history["epoch"][best_idx]
    best_map = history["mAP50"][best_idx]
    ax1.annotate(f'Best: {best_map:.3f}', xy=(best_epoch, best_map),
                 xytext=(best_epoch + 2, best_map - 0.05),
                 fontsize=9, ha='left',
                 arrowprops=dict(arrowstyle='->', color='gray'))
    
    # 2. Precision & Recall
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history["epoch"], history["precision"], 'b-', linewidth=2,
             label="Precision", alpha=0.8)
    ax2.plot(history["epoch"], history["recall"], 'r-', linewidth=2,
             label="Recall", alpha=0.8)
    ax2.set_title("Precision & Recall", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(1, max(history["epoch"]))
    
    # 3. Training Losses
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history["epoch"], history["box_loss"], label="Box Loss", 
             linewidth=2, alpha=0.8, color='#e74c3c')
    ax3.plot(history["epoch"], history["cls_loss"], label="Cls Loss", 
             linewidth=2, alpha=0.8, color='#3498db')
    ax3.plot(history["epoch"], history["dfl_loss"], label="DFL Loss", 
             linewidth=2, alpha=0.8, color='#27ae60')
    ax3.set_title("Training Losses", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Loss", fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_xlim(1, max(history["epoch"]))
    
    # 4. Combined Loss (smoothed)
    ax4 = fig.add_subplot(gs[1, 0])
    total_loss = [b + c + d for b, c, d in 
                  zip(history["box_loss"], history["cls_loss"], history["dfl_loss"])]
    
    # Simple moving average for smoothing
    window = min(5, len(total_loss))
    if window > 1:
        smoothed = np.convolve(total_loss, np.ones(window)/window, mode='valid')
        smooth_epochs = history["epoch"][window-1:]
    else:
        smoothed = total_loss
        smooth_epochs = history["epoch"]
    
    ax4.plot(history["epoch"], total_loss, 'gray', alpha=0.3, linewidth=1, label="Raw")
    ax4.plot(smooth_epochs, smoothed, 'navy', linewidth=2.5, label="Smoothed")
    ax4.set_title("Total Loss (Smoothed)", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Epoch", fontsize=11)
    ax4.set_ylabel("Total Loss", fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    ax4.set_xlim(1, max(history["epoch"]))
    
    # 5. F1 Score approximation
    ax5 = fig.add_subplot(gs[1, 1])
    f1_scores = []
    for p, r in zip(history["precision"], history["recall"]):
        if p + r > 0:
            f1_scores.append(2 * p * r / (p + r))
        else:
            f1_scores.append(0)
    
    ax5.plot(history["epoch"], f1_scores, 'darkorange', linewidth=2.5,
             marker='o', markersize=3)
    ax5.fill_between(history["epoch"], 0, f1_scores, alpha=0.2, color='orange')
    ax5.set_title("F1 Score", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Epoch", fontsize=11)
    ax5.set_ylabel("F1", fontsize=11)
    ax5.grid(alpha=0.3)
    ax5.set_ylim(0, 1)
    ax5.set_xlim(1, max(history["epoch"]))
    
    # 6. Training Summary Stats
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Current stats text
    stats_text = f"""
    TRAINING PROGRESS - Epoch {epoch}/{EPOCHS}
    ═══════════════════════════════════════════
    
    Current Metrics:
    ───────────────────────────────────────────
    mAP@0.5       : {history['mAP50'][-1]:.4f}
    mAP@0.5:0.95  : {history['mAP50_95'][-1]:.4f}
    Precision     : {history['precision'][-1]:.4f}
    Recall        : {history['recall'][-1]:.4f}
    F1 Score      : {f1_scores[-1]:.4f}
    
    Best So Far:
    ───────────────────────────────────────────
    Best mAP@0.5  : {max(history['mAP50']):.4f} (Epoch {best_epoch})
    
    Current Losses:
    ───────────────────────────────────────────
    Box Loss      : {history['box_loss'][-1]:.4f}
    Cls Loss      : {history['cls_loss'][-1]:.4f}
    DFL Loss      : {history['dfl_loss'][-1]:.4f}
    """
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    # Main title
    fig.suptitle("YOLOv11x Bone Fracture Detection - Training Progress",
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_final_summary(history: dict, final_metrics: dict, class_aps: dict, 
                        output_path: Path):
    """Create comprehensive final training summary visualization."""
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. mAP Progress (full training)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(history["epoch"], history["mAP50"], 'g-', linewidth=3,
             label="mAP@0.5", alpha=0.9)
    ax1.plot(history["epoch"], history["mAP50_95"], 'purple', linewidth=3,
             label="mAP@0.5:0.95", alpha=0.9)
    ax1.fill_between(history["epoch"], 0, history["mAP50"], alpha=0.15, color='green')
    ax1.axhline(y=final_metrics["mAP50"], color='g', linestyle='--', alpha=0.5)
    ax1.set_title("Validation mAP Throughout Training", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("mAP", fontsize=12)
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Per-Class AP Bar Chart
    ax2 = fig.add_subplot(gs[0, 2])
    classes = list(class_aps.keys())
    aps = list(class_aps.values())
    
    # Sort by AP
    sorted_idx = np.argsort(aps)[::-1]
    classes = [classes[i] for i in sorted_idx]
    aps = [aps[i] for i in sorted_idx]
    colors = [CLASS_COLORS[CLASS_NAMES.index(c)] for c in classes]
    
    bars = ax2.barh(range(len(classes)), aps, color=colors, edgecolor='white', linewidth=1.5)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(classes, fontsize=11)
    ax2.set_xlabel("AP@0.5", fontsize=12)
    ax2.set_title("Per-Class AP@0.5", fontsize=16, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, ap in zip(bars, aps):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{ap:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # 3. Loss Curves
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(history["epoch"], history["box_loss"], label="Box", linewidth=2, color='#e74c3c')
    ax3.plot(history["epoch"], history["cls_loss"], label="Cls", linewidth=2, color='#3498db')
    ax3.plot(history["epoch"], history["dfl_loss"], label="DFL", linewidth=2, color='#27ae60')
    ax3.set_title("Training Losses", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Precision/Recall
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(history["epoch"], history["precision"], 'b-', linewidth=2, label="Precision")
    ax4.plot(history["epoch"], history["recall"], 'r-', linewidth=2, label="Recall")
    ax4.set_title("Precision & Recall", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Score")
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # 5. F1 Score
    ax5 = fig.add_subplot(gs[1, 2])
    f1_scores = []
    for p, r in zip(history["precision"], history["recall"]):
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores.append(f1)
    ax5.plot(history["epoch"], f1_scores, 'darkorange', linewidth=2.5)
    ax5.fill_between(history["epoch"], 0, f1_scores, alpha=0.2, color='orange')
    ax5.set_title("F1 Score Progress", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("F1")
    ax5.grid(alpha=0.3)
    ax5.set_ylim(0, 1)
    
    # 6. Final Metrics Summary (large panel)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                           FINAL TRAINING RESULTS - GrazPedWri-DX Bone Fracture Detection                  ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                           ║
    ║   Model: YOLOv11x                            Image Size: {IMGSZ}px                                         ║
    ║   Total Epochs: {len(history['epoch'])}                             Batch Size: {BATCH}                                           ║
    ║                                                                                                           ║
    ╠════════════════════════════════════════════════════════════════╦══════════════════════════════════════════╣
    ║                     DETECTION METRICS                          ║              BEST EPOCH                   ║
    ╠════════════════════════════════════════════════════════════════╬══════════════════════════════════════════╣
    ║   mAP@0.5       :  {final_metrics['mAP50']:.4f}                                     ║   Epoch {np.argmax(history['mAP50']) + 1}                              ║
    ║   mAP@0.5:0.95  :  {final_metrics['mAP50_95']:.4f}                                     ║   Best mAP@0.5: {max(history['mAP50']):.4f}                   ║
    ║   Precision     :  {final_metrics['precision']:.4f}                                     ║                                           ║
    ║   Recall        :  {final_metrics['recall']:.4f}                                     ║                                           ║
    ╚════════════════════════════════════════════════════════════════╩══════════════════════════════════════════╝
    """
    
    ax6.text(0.5, 0.6, summary_text, transform=ax6.transAxes,
             fontsize=12, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor='#16213e',
                      alpha=0.95))
    ax6.text(0.5, 0.6, summary_text.replace('█', ' '), transform=ax6.transAxes,
             fontsize=12, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', color='#00ff88')
    
    # Main title
    fig.suptitle("🦴 YOLOv11x Bone Fracture Detection - FINAL TRAINING SUMMARY 🦴",
                 fontsize=22, fontweight='bold', y=0.98, color='#1a1a2e')
    
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()


# ==================== TRAINING CALLBACK ====================
def on_train_epoch_end(trainer):
    """Callback executed at end of each training epoch."""
    epoch = trainer.epoch + 1
    
    # Extract validation metrics
    metrics = trainer.metrics
    mAP50 = float(metrics.get("metrics/mAP50(B)", 0.0))
    mAP = float(metrics.get("metrics/mAP50-95(B)", 0.0))
    precision = float(metrics.get("metrics/precision(B)", 0.0))
    recall = float(metrics.get("metrics/recall(B)", 0.0))
    
    # Extract losses
    losses = trainer.loss_items
    if losses is not None and len(losses) >= 3:
        box_l = float(losses[0].item())
        cls_l = float(losses[1].item())
        dfl_l = float(losses[2].item())
    else:
        box_l = cls_l = dfl_l = 0.0
    
    # Save to history
    history["epoch"].append(epoch)
    history["mAP50"].append(mAP50)
    history["mAP50_95"].append(mAP)
    history["precision"].append(precision)
    history["recall"].append(recall)
    history["box_loss"].append(box_l)
    history["cls_loss"].append(cls_l)
    history["dfl_loss"].append(dfl_l)
    
    # Print status
    status = "🔥 Excellent" if mAP50 > 0.80 else "✨ Great" if mAP50 > 0.70 else "📈 Good" if mAP50 > 0.60 else "🔄 Training"
    
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch}/{EPOCHS} - VALIDATION RESULTS {status}")
    print(f"{'='*80}")
    print(f"  mAP@0.5       : {mAP50:.4f}")
    print(f"  mAP@0.5:0.95  : {mAP:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  Losses → Box: {box_l:.4f} | Cls: {cls_l:.4f} | DFL: {dfl_l:.4f}")
    
    # ==================== PER-CLASS METRICS ====================
    # Try to get per-class AP from validator
    try:
        validator = trainer.validator
        if validator is not None and hasattr(validator, 'metrics'):
            val_metrics = validator.metrics
            if hasattr(val_metrics, 'box') and hasattr(val_metrics.box, 'ap50'):
                ap50_per_class = val_metrics.box.ap50
                names = validator.names if hasattr(validator, 'names') else CLASS_NAMES
                
                print(f"\n  {'─'*40}")
                print(f"  PER-CLASS AP@0.5:")
                print(f"  {'─'*40}")
                
                for i, cls_name in enumerate(CLASS_NAMES):
                    if i < len(ap50_per_class):
                        ap = float(ap50_per_class[i])
                        bar = "█" * int(ap * 15)
                        bar_empty = "░" * (15 - int(ap * 15))
                        print(f"  {cls_name:<18}: {ap:.4f} {bar}{bar_empty}")
                
                print(f"  {'─'*40}")
    except Exception as e:
        # Silently skip if per-class metrics not available
        pass
    
    print(f"{'='*80}\n")
    
    # Create progress plot
    create_progress_plot(history, OUTPUT_DIR / "live_progress.png", epoch)
    
    # Save epoch snapshot every 10 epochs
    if epoch % 10 == 0:
        create_progress_plot(history, PLOTS_DIR / f"epoch_{epoch:03d}.png", epoch)
    
    # Save history to JSON
    with open(OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)



# ==================== MAIN TRAINING ====================
def main():
    print("\n" + "█" * 80)
    print("BONE FRACTURE DETECTION - YOLOv11x TRAINING")
    print("█" * 80)
    print(f"\nDataset    : {DATA_YAML}")
    print(f"Model      : {MODEL_NAME}")
    print(f"Image Size : {IMGSZ}")
    print(f"Batch Size : {BATCH}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Output     : {OUTPUT_DIR}")
    print("█" * 80 + "\n")
    
    # Check if data.yaml exists
    if not Path(DATA_YAML).exists():
        print(f"⚠️  ERROR: Data config not found at {DATA_YAML}")
        print("Please run preprocessing first:")
        print("  python scripts/preprocess.py --input /path/to/grazpedwri --output ./dataset")
        return
    
    # Load model
    model = YOLO(MODEL_NAME)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    # Start training
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        
        # Optimizer settings
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.008,
        weight_decay=0.0005,
        warmup_epochs=5,
        freeze=10,  # Freeze first 10 layers initially
        
        # Augmentations (tuned for medical X-ray images)
        hsv_h=0.01,      # Minimal hue variation (X-rays are grayscale)
        hsv_s=0.3,       # Saturation variation
        hsv_v=0.4,       # Value/brightness variation
        degrees=15,      # Rotation
        translate=0.15,  # Translation
        scale=0.5,       # Scale variation
        shear=3.0,       # Shear
        flipud=0.5,      # Vertical flip
        fliplr=0.5,      # Horizontal flip
        mosaic=1.0,      # Mosaic augmentation
        mixup=0.3,       # Mixup augmentation
        copy_paste=0.3,  # Copy-paste augmentation
        close_mosaic=15, # Disable mosaic for last 15 epochs
        
        # Loss weights
        box=8.0,
        cls=0.5,
        dfl=1.5,
        
        # Other settings
        cache="ram",
        device=DEVICE,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True,
        save_period=20,
    )
    
    # ==================== FINAL EVALUATION WITH TTA ====================
    print("\n" + "="*80)
    print("FINAL EVALUATION WITH TEST-TIME AUGMENTATION (TTA)")
    print("="*80)
    
    best_pt = OUTPUT_DIR / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(str(results.save_dir)) / "weights" / "best.pt"
    
    print(f"Loading best model: {best_pt}")
    model = YOLO(str(best_pt))
    
    final = model.val(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        batch=1,
        augment=True,  # TTA
        conf=0.001,
        iou=0.6,
        device=DEVICE,
        save_json=True,
        plots=True
    )
    
    # ==================== GENERATE FINAL REPORT ====================
    final_metrics = {
        "mAP50": round(float(final.box.map50), 4),
        "mAP50_95": round(float(final.box.map), 4),
        "precision": round(float(final.box.mp), 4),
        "recall": round(float(final.box.mr), 4),
    }
    
    # Per-class AP
    class_aps = {}
    for i, name in enumerate(final.names.values()):
        if i < len(final.box.ap50):
            class_aps[name] = round(float(final.box.ap50[i]), 4)
    
    # Save final report
    report = {
        "run_name": RUN_NAME,
        "model": MODEL_NAME,
        "epochs_trained": len(history["epoch"]),
        "image_size": IMGSZ,
        "batch_size": BATCH,
        "final_metrics": final_metrics,
        "per_class_AP50": class_aps,
        "best_mAP50": round(max(history["mAP50"]), 4),
        "best_epoch": int(np.argmax(history["mAP50"]) + 1),
        "best_model_path": str(best_pt),
    }
    
    with open(OUTPUT_DIR / "FINAL_REPORT.json", 'w') as f:
        json.dump(report, f, indent=4)
    
    # Create final summary visualization
    create_final_summary(
        history, final_metrics, class_aps,
        OUTPUT_DIR / "FINAL_TRAINING_SUMMARY.png"
    )
    
    # ==================== PRINT FINAL RESULTS ====================
    print("\n" + "█" * 80)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("█" * 80)
    print(f"\n  Final mAP@0.5      : {final_metrics['mAP50']:.4f}")
    print(f"  Final mAP@0.5:0.95 : {final_metrics['mAP50_95']:.4f}")
    print(f"  Precision          : {final_metrics['precision']:.4f}")
    print(f"  Recall             : {final_metrics['recall']:.4f}")
    print(f"\n  Per-Class AP@0.5:")
    for cls, ap in sorted(class_aps.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(ap * 20)
        print(f"    {cls:<20}: {ap:.4f} {bar}")
    print(f"\n  Best model         : {best_pt}")
    print(f"  Report             : {OUTPUT_DIR / 'FINAL_REPORT.json'}")
    print(f"  Summary plot       : {OUTPUT_DIR / 'FINAL_TRAINING_SUMMARY.png'}")
    print(f"  Progress plots     : {PLOTS_DIR}")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()
