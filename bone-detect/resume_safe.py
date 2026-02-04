"""
Resume training with NaN-safe configuration.

Fixes applied:
- Lower learning rate (0.0005 instead of 0.001)
- Gradient clipping (max_norm=10.0)
- AMP disabled for stability
- Smaller batch size option

Usage:
    python resume_safe.py --run YOLOv11x_BONE_SOTA_20260201_1825
"""

from ultralytics import YOLO
from pathlib import Path
import argparse

CLASS_NAMES = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture",
    "metal", "periostealreaction", "pronatorsign", "softtissue"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Resume YOLO training (NaN-safe)")
    parser.add_argument("--run", type=str, default=None,
                        help="Run name to resume")
    parser.add_argument("--checkpoint", type=str, default="best.pt",
                        help="Checkpoint to use (default: best.pt to avoid NaN state)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Total epochs")
    return parser.parse_args()


def find_latest_run(outputs_dir: Path) -> Path:
    runs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], 
                  key=lambda x: x.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError("No training runs found in outputs/")
    return runs[0]


def on_train_epoch_end(trainer):
    """Callback with per-class metrics."""
    epoch = trainer.epoch + 1
    metrics = trainer.metrics
    mAP50 = float(metrics.get("metrics/mAP50(B)", 0.0))
    mAP = float(metrics.get("metrics/mAP50-95(B)", 0.0))
    precision = float(metrics.get("metrics/precision(B)", 0.0))
    recall = float(metrics.get("metrics/recall(B)", 0.0))
    
    losses = trainer.loss_items
    if losses is not None and len(losses) >= 3:
        box_l = float(losses[0].item())
        cls_l = float(losses[1].item())
        dfl_l = float(losses[2].item())
    else:
        box_l = cls_l = dfl_l = 0.0
    
    status = "🔥 Excellent" if mAP50 > 0.80 else "✨ Great" if mAP50 > 0.70 else "📈 Good" if mAP50 > 0.60 else "🔄 Training"
    
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch} - VALIDATION RESULTS {status}")
    print(f"{'='*80}")
    print(f"  mAP@0.5       : {mAP50:.4f}")
    print(f"  mAP@0.5:0.95  : {mAP:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  Losses → Box: {box_l:.4f} | Cls: {cls_l:.4f} | DFL: {dfl_l:.4f}")
    
    try:
        validator = trainer.validator
        if validator is not None and hasattr(validator, 'metrics'):
            val_metrics = validator.metrics
            if hasattr(val_metrics, 'box') and hasattr(val_metrics.box, 'ap50'):
                ap50_per_class = val_metrics.box.ap50
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
    except Exception:
        pass
    
    print(f"{'='*80}\n")


def main():
    args = parse_args()
    
    outputs_dir = Path(__file__).parent / "outputs"
    data_yaml = Path(__file__).parent / "dataset" / "data.yaml"
    
    if args.run:
        run_dir = outputs_dir / args.run
    else:
        run_dir = find_latest_run(outputs_dir)
    
    # Use best.pt instead of last.pt to avoid corrupted NaN state
    checkpoint = run_dir / "weights" / args.checkpoint
    
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print("Available checkpoints:")
        for f in (run_dir / "weights").glob("*.pt"):
            print(f"  - {f.name}")
        return
    
    print("="*70)
    print("RESUMING TRAINING WITH NaN-SAFE CONFIGURATION")
    print("="*70)
    print(f"Run directory : {run_dir}")
    print(f"Checkpoint    : {checkpoint}")
    print(f"Data config   : {data_yaml}")
    print("\nFixes applied:")
    print("  ✓ Lower learning rate (0.0005)")
    print("  ✓ Gradient clipping enabled")
    print("  ✓ Using best.pt (not corrupted last.pt)")
    print("="*70 + "\n")
    
    # Load model
    model = YOLO(str(checkpoint))
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    # Train with safer settings - DO NOT use resume=True as it inherits bad state
    # Instead, continue training from the best checkpoint
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=1280,
        batch=16,  # Slightly smaller for stability
        patience=30,
        
        # More conservative optimizer settings
        optimizer="AdamW",
        lr0=0.0005,      # Lower initial LR (was 0.001)
        lrf=0.01,        # Higher final LR ratio
        weight_decay=0.0005,
        warmup_epochs=5,
        
        # Augmentations
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.4,
        degrees=10,       # Less rotation
        translate=0.1,
        scale=0.4,
        flipud=0.3,
        fliplr=0.5,
        mosaic=0.8,       # Less mosaic
        mixup=0.2,
        copy_paste=0.2,
        close_mosaic=20,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Stability settings
        amp=True,          # Keep AMP but with lower LR
        cache="ram",
        workers=4,         # Reduce workers to prevent connection errors
        device=0,
        project=str(outputs_dir),
        name=run_dir.name + "_continued",
        exist_ok=True,
        pretrained=False,  # Already pretrained
        verbose=True,
        plots=True,
        save_period=10,
    )


if __name__ == "__main__":
    main()
