"""
Resume training from a crashed/interrupted run with per-class metrics.

Usage:
    python resume.py --run YOLOv11x_BONE_SOTA_20260201_1825
    
Or to resume the most recent run:
    python resume.py
"""

from ultralytics import YOLO
from pathlib import Path
import argparse

# Class names for per-class metrics
CLASS_NAMES = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture",
    "metal", "periostealreaction", "pronatorsign", "softtissue"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Resume YOLO training")
    parser.add_argument("--run", type=str, default=None,
                        help="Run name to resume (default: most recent)")
    parser.add_argument("--checkpoint", type=str, default="best.pt",
                        help="Checkpoint to resume from (default: best.pt)")
    return parser.parse_args()


def find_latest_run(outputs_dir: Path) -> Path:
    """Find the most recent training run."""
    runs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], 
                  key=lambda x: x.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError("No training runs found in outputs/")
    return runs[0]


def on_train_epoch_end(trainer):
    """Callback to print per-class metrics after each epoch."""
    epoch = trainer.epoch + 1
    
    # Get overall metrics
    metrics = trainer.metrics
    mAP50 = float(metrics.get("metrics/mAP50(B)", 0.0))
    mAP = float(metrics.get("metrics/mAP50-95(B)", 0.0))
    precision = float(metrics.get("metrics/precision(B)", 0.0))
    recall = float(metrics.get("metrics/recall(B)", 0.0))
    
    # Get losses
    losses = trainer.loss_items
    if losses is not None and len(losses) >= 3:
        box_l = float(losses[0].item())
        cls_l = float(losses[1].item())
        dfl_l = float(losses[2].item())
    else:
        box_l = cls_l = dfl_l = 0.0
    
    # Print status
    status = "🔥 Excellent" if mAP50 > 0.80 else "✨ Great" if mAP50 > 0.70 else "📈 Good" if mAP50 > 0.60 else "🔄 Training"
    
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch} - VALIDATION RESULTS {status}")
    print(f"{'='*80}")
    print(f"  mAP@0.5       : {mAP50:.4f}")
    print(f"  mAP@0.5:0.95  : {mAP:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  Losses → Box: {box_l:.4f} | Cls: {cls_l:.4f} | DFL: {dfl_l:.4f}")
    
    # ==================== PER-CLASS METRICS ====================
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
    
    if args.run:
        run_dir = outputs_dir / args.run
    else:
        run_dir = find_latest_run(outputs_dir)
    
    checkpoint = run_dir / "weights" / args.checkpoint
    
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print(f"Available checkpoints:")
        for f in (run_dir / "weights").glob("*.pt"):
            print(f"  - {f.name}")
        return
    
    print("="*60)
    print("RESUMING TRAINING WITH PER-CLASS METRICS")
    print("="*60)
    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {checkpoint}")
    print("="*60 + "\n")
    
    # Load model and add callback
    model = YOLO(str(checkpoint))
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    # Resume training
    model.train(resume=True)


if __name__ == "__main__":
    main()
