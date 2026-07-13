"""
Classification trainer for dental X-ray crops.
Uses crops generated from YOLO detection pipeline.
Includes comprehensive reporting for presentation.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # Oral-Dental/oral-xray/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import timm

from src.models.classification import (
    ClassificationModel, 
    CLASSIFICATION_MODELS,
    DENTAL_RECOMMENDED_MODELS
)
from src.utils.classification_plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_class_distribution,
    save_sample_predictions,
    generate_classification_report
)


def get_transforms(input_size: int, is_train: bool = True):
    """Get data transforms for training/validation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:  # Mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss/len(loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate model and return metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        total_loss / len(loader), 
        100. * correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )


def compute_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int, class_names: list):
    """Compute confusion matrix and per-class metrics."""
    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for pred, label in zip(preds, labels):
        cm[label, pred] += 1
    
    # Per-class metrics
    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, :].sum()
        
        per_class[name] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': int(support)
        }
    
    return cm, per_class


def get_class_counts(dataset_path: Path) -> dict:
    """Count samples per class from ImageFolder directory."""
    counts = {}
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            # Extract actual class name (remove index prefix like "0_Apical_Periodontitis")
            if '_' in class_name and class_name.split('_')[0].isdigit():
                class_name = ' '.join(class_name.split('_')[1:])
            count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
            counts[class_name] = count
    return counts


def main():
    parser = argparse.ArgumentParser(description="Train classification model on dental crops")
    parser.add_argument("--crops_root", required=True, help="Path to crops directory")
    parser.add_argument("--model", default="convnext_large", 
                        choices=list(CLASSIFICATION_MODELS.keys()),
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Epochs to train with frozen backbone")
    parser.add_argument("--output_dir", default="Oral-Dental/oral-xray/outputs/classification")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    
    # Setup
    crops_root = Path(args.crops_root)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Load class info
    classes_json = crops_root / "classes.json"
    if classes_json.exists():
        class_info = json.loads(classes_json.read_text())
        num_classes = class_info["nc"]
        class_names = class_info["names"]
    else:
        # Infer from directories
        train_dir = crops_root / "train"
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        num_classes = len(class_names)
    
    print(f"📊 Classes ({num_classes}): {class_names}")
    
    # Get model config
    model_cfg = CLASSIFICATION_MODELS[args.model]
    input_size = model_cfg['input_size']
    model_name = model_cfg['name']
    
    # Create datasets
    train_transform = get_transforms(input_size, is_train=True)
    val_transform = get_transforms(input_size, is_train=False)
    
    train_dataset = datasets.ImageFolder(crops_root / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(crops_root / "val", transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    print(f"📁 Train: {len(train_dataset)} samples")
    print(f"📁 Val: {len(val_dataset)} samples")
    
    # Get class counts for reporting
    train_counts = get_class_counts(crops_root / "train")
    val_counts = get_class_counts(crops_root / "val")
    
    # Create model
    model = ClassificationModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        dropout=args.dropout,
        multi_label=False
    )
    model = model.to(device)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.model}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save weights directory
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"\n🚀 Training {args.model} for {args.epochs} epochs")
    print(f"   Input size: {input_size}x{input_size}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Output: {output_dir}")
    
    # Phase 1: Frozen backbone
    if args.freeze_epochs > 0:
        print(f"\n❄️ Phase 1: Training classifier only ({args.freeze_epochs} epochs)")
        model.freeze_backbone()
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr * 10,  # Higher LR for classifier only
            weight_decay=args.weight_decay
        )
        
        for epoch in range(args.freeze_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{args.freeze_epochs} | "
                  f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
                  f"Val: {val_loss:.4f} / {val_acc:.2f}%")
    
    # Phase 2: Full fine-tuning
    print(f"\n🔥 Phase 2: Full fine-tuning ({args.epochs} epochs)")
    model.unfreeze_backbone()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    best_preds = None
    best_labels = None
    best_probs = None
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, preds, labels, probs = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Log
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
              f"Val: {val_loss:.4f} / {val_acc:.2f}% | "
              f"LR: {lr:.2e}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            best_preds = preds
            best_labels = labels
            best_probs = probs
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names,
                'model_name': args.model,
                'input_size': input_size,
            }, weights_dir / "best.pt")
            print(f"   💾 Saved best model: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Save final
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
        'class_names': class_names,
        'model_name': args.model,
        'input_size': input_size,
    }, weights_dir / "last.pt")
    
    # ============ GENERATE REPORTS ============
    print("\n📊 Generating reports...")
    
    # Compute final metrics
    cm, per_class_metrics = compute_metrics(best_preds, best_labels, num_classes, class_names)
    
    # Save history
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    
    # Save config
    config = vars(args)
    config['best_val_acc'] = best_val_acc
    config['best_epoch'] = best_epoch
    config['num_classes'] = num_classes
    config['class_names'] = class_names
    config['input_size'] = input_size
    config['model_name_timm'] = model_name
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    
    # Generate full report
    report_dir = generate_classification_report(
        history=history,
        confusion_matrix=cm,
        per_class_metrics=per_class_metrics,
        class_names=class_names,
        train_counts=train_counts,
        val_counts=val_counts,
        config=config,
        out_dir=output_dir,
        best_epoch=best_epoch,
        best_acc=best_val_acc
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"📈 Best Validation Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"\n📊 Per-Class Performance:")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 65)
    for name, metrics in per_class_metrics.items():
        print(f"{name:<25} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
              f"{metrics['f1']:>10.3f} {metrics['support']:>10d}")
    
    # Macro averages
    macro_p = np.mean([m['precision'] for m in per_class_metrics.values()])
    macro_r = np.mean([m['recall'] for m in per_class_metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in per_class_metrics.values()])
    print("-" * 65)
    print(f"{'Macro Average':<25} {macro_p:>10.3f} {macro_r:>10.3f} {macro_f1:>10.3f}")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📁 Presentation folder: {report_dir}")


if __name__ == "__main__":
    main()
