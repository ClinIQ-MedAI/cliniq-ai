#!/usr/bin/env python3
"""
Training Script for NIH Chest X-ray Multi-Label Classification

Features:
- ConvNeXt-Large backbone
- Focal Loss with per-class weights
- Mixed Precision Training (AMP)
- Gradient Accumulation
- EMA weights
- Cosine Annealing scheduler
- AUC-ROC as primary metric
"""

import os
import sys
import time
import json
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from data.dataset import get_dataloaders
from models.convnext import ConvNeXtClassifier, EMA
from utils.losses import FocalLoss, create_loss
from utils.metrics import MetricTracker, compute_auc_roc, find_optimal_thresholds


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 2,
    ema: EMA = None,
    scheduler = None  # Added scheduler
) -> float:
    """Train for one epoch with gradient accumulation and AMP."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Mixed precision forward
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps  # Scale loss for accumulation
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Step Scheduler per update (NOT per batch, per optimizer step)
            if scheduler is not None:
                scheduler.step()
            
            # Update EMA
            if ema is not None:
                ema.update()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    class_names: list
) -> tuple:
    """Validate model and compute metrics."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    metric_tracker = MetricTracker(class_names)
    
    for images, targets in tqdm(loader, desc='Validating', leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Collect predictions (apply sigmoid for probabilities)
        probs = torch.sigmoid(outputs)
        metric_tracker.update(probs, targets)
    
    # Compute metrics
    # Compute metrics
    metrics = metric_tracker.compute()
    avg_loss = total_loss / num_batches
    
    # Return collected predictions and targets
    all_probs = np.concatenate(metric_tracker.all_preds)
    all_targets = np.concatenate(metric_tracker.all_targets)
    
    return avg_loss, metrics, all_probs, all_targets


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_auc: float,
    thresholds: dict,
    filepath: str,
    ema: EMA = None
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_auc': best_auc,
        'thresholds': thresholds,
        'config': {
            'model_name': config.model.name,
            'num_classes': config.model.num_classes,
            'image_size': config.training.image_size,
            'classes': config.data.classes
        }
    }
    
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow
    
    torch.save(checkpoint, filepath)


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train ConvNeXt for Chest X-ray Classification')
    parser.add_argument('--resume', nargs='?', const='last.pt', help='Path to checkpoint to resume from (default: last.pt in checkpoint_dir)')
    args = parser.parse_args()

    # Setup
    cfg = config
    logger = setup_logging(cfg.output.log_dir)
    set_seed(cfg.hardware.seed)
    
    logger.info("=" * 60)
    logger.info("NIH CHEST X-RAY MULTI-LABEL CLASSIFICATION")
    logger.info("=" * 60)
    logger.info(f"\n{cfg}")
    
    # Device
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Data
    logger.info("\nLoading data...")
    train_loader, val_loader, _, train_dataset = get_dataloaders(
        batch_size=cfg.training.batch_size,
        image_size=cfg.training.image_size,
        num_workers=cfg.hardware.num_workers,
        use_official_split=True,
        seed=cfg.hardware.seed
    )
    
    class_names = cfg.data.classes
    num_classes = len(class_names)
    
    # Model
    logger.info(f"\nCreating model: {cfg.model.name}")
    model = ConvNeXtClassifier(
        num_classes=num_classes,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # EMA
    ema = None
    if cfg.ema.enabled:
        ema = EMA(model, decay=cfg.ema.decay)
        logger.info(f"EMA enabled with decay={cfg.ema.decay}")
    
    # Loss with per-class alpha weights
    alpha = torch.tensor(train_dataset.class_weights)
    criterion = create_loss(
        loss_type=cfg.training.loss_type,
        alpha=alpha,
        pos_weight=torch.tensor(train_dataset.pos_weights),
        gamma=cfg.training.focal_gamma,
        label_smoothing=cfg.training.label_smoothing
    )
    logger.info(f"Loss: {cfg.training.loss_type} (gamma={cfg.training.focal_gamma})")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    # Scheduler with warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=cfg.training.warmup_epochs * len(train_loader)
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(cfg.training.epochs - cfg.training.warmup_epochs) * len(train_loader),
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[cfg.training.warmup_epochs * len(train_loader)]
    )
    logger.info(f"Scheduler: Cosine Annealing with {cfg.training.warmup_epochs} warmup epochs")
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=cfg.hardware.mixed_precision)
    logger.info(f"Mixed Precision: {cfg.hardware.mixed_precision}")
    
    # Output directories
    os.makedirs(cfg.output.checkpoint_dir, exist_ok=True)
    
    # Initialize variables
    start_epoch = 0
    best_auc = 0
    best_thresholds = {name: 0.5 for name in class_names}
    
    # Resume from checkpoint
    if args.resume:
        # Construct path if only filename or const is provided
        if os.path.dirname(args.resume) == '':
            checkpoint_path = os.path.join(cfg.output.checkpoint_dir, args.resume)
        else:
            checkpoint_path = args.resume
            
        if os.path.isfile(checkpoint_path):
            logger.info(f"\nResuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Load states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if ema is not None and 'ema_shadow' in checkpoint:
                ema.shadow = checkpoint['ema_shadow']
                logger.info("Restored EMA state")
                
            start_epoch = checkpoint['epoch'] + 1
            best_auc = checkpoint.get('best_auc', 0)
            best_thresholds = checkpoint.get('thresholds', best_thresholds)
            
            logger.info(f"Resumed from epoch {start_epoch} (Best AUC: {best_auc:.4f})")
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}, starting from scratch.")

    # Training loop
    patience_counter = 0
    
    logger.info(f"\nStarting training for {cfg.training.epochs} epochs (starting from {start_epoch})...")
    logger.info(f"Gradient accumulation steps: {cfg.training.gradient_accumulation_steps}")
    
    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_start = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            accumulation_steps=cfg.training.gradient_accumulation_steps,
            ema=ema,
            scheduler=scheduler
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validate (with EMA if available)
        if ema is not None:
            ema.apply_shadow()
        
        # Optimized: validate now returns raw predictions too
        val_loss, metrics, val_probs, val_targets = validate(model, val_loader, criterion, device, class_names)
        
        if ema is not None:
            ema.restore()
        
        # Log metrics
        mean_auc = metrics['mean_auc']
        epoch_time = time.time() - epoch_start
        
        logger.info(f"\nTrain Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Mean AUC-ROC: {mean_auc:.4f}")
        logger.info(f"Mean PR-AUC: {metrics.get('mean_pr_auc', 0):.4f}")
        logger.info(f"LR: {current_lr:.6f}")
        logger.info(f"Time: {epoch_time:.1f}s")
        
        # Per-class AUC
        logger.info("\nPer-class AUC-ROC:")
        for name in class_names:
            auc = metrics.get(name, 0)
            logger.info(f"  {name}: {auc:.4f}")
        
        # Save best model
        if mean_auc > best_auc:
            best_auc = mean_auc
            patience_counter = 0
            
            logger.info("Computing optimal thresholds (Instant)...")
            
            try:
                from sklearn.metrics import precision_recall_curve
                
                # Calculate best threshold per class (Maximize F1) using cached validation results
                best_thresholds = {}
                
                # val_probs might optionally need sigmoid if validate didn't apply it or applied it differently
                # In validate(), we did: probs = torch.sigmoid(outputs) -> MetricTracker
                # So val_probs are already probabilities.
                
                for i, name in enumerate(class_names):
                    # val_targets and val_probs are already numpy arrays from validate() return
                    precision, recall, thresholds_pr = precision_recall_curve(val_targets[:, i], val_probs[:, i])
                    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
                    best_idx = np.argmax(f1_scores)
                    best_thresholds[name] = float(thresholds_pr[best_idx]) if len(thresholds_pr) > best_idx else 0.5
                    
                logger.info(f"Optimal Thresholds: {json.dumps(best_thresholds, indent=2)}")
            except ImportError:
                logger.warning("sklearn not found - using default 0.5 thresholds")
            except Exception as e:
                logger.error(f"Error calculating thresholds: {e}")
                logger.warning("Using default 0.5 thresholds due to error")
            
            # Save checkpoint
            checkpoint_path = os.path.join(cfg.output.checkpoint_dir, 'best.pt')
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_auc,
                best_thresholds, checkpoint_path, ema
            )
            logger.info(f"✅ Saved best model with AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
        
        # Save last checkpoint
        last_path = os.path.join(cfg.output.checkpoint_dir, 'last.pt')
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_auc,
            best_thresholds, last_path, ema
        )
        
        # Early stopping
        if patience_counter >= cfg.training.early_stopping_patience:
            logger.info(f"\n⚠️ Early stopping after {epoch+1} epochs")
            break
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Best Mean AUC-ROC: {best_auc:.4f}")
    logger.info(f"Checkpoints saved to: {cfg.output.checkpoint_dir}")


if __name__ == '__main__':
    main()
