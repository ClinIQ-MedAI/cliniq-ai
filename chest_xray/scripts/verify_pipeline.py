#!/usr/bin/env python3
"""
Verification Script for Chest X-ray Pipeline
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import traceback

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from data.dataset import get_dataloaders
from models.convnext import ConvNeXtClassifier
from utils.losses import create_loss

def verify_pipeline():
    print("=" * 60, flush=True)
    print("PIPELINE VERIFICATION", flush=True)
    print("=" * 60, flush=True)
    
    # 1. Verify Configuration
    print("\n1. Checking Configuration...", flush=True)
    try:
        print(f"   Model: {config.model.name}")
        print(f"   Image Size: {config.training.image_size}")
        print(f"   Batch Size: {config.training.batch_size}")
        print(f"   Device: {config.hardware.device}")
        print("   ✅ Config loaded")
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return

    # 2. Verify Data Loading
    print("\n2. Checking Data Loading...")
    try:
        train_loader, val_loader, test_loader, train_dataset = get_dataloaders(
            batch_size=config.training.batch_size,
            image_size=config.training.image_size,
            num_workers=config.hardware.num_workers,
            use_official_split=True
        )
        print(f"   Train images: {len(train_dataset)}")
        print(f"   Train batches: {len(train_loader)}")
        
        # Get a batch
        images, targets = next(iter(train_loader))
        print(f"   Batch shape: {images.shape}")
        print(f"   Targets shape: {targets.shape}")
        
        if images.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {images.shape[1]}")
            
        print("   ✅ Data loading successful")
    except Exception as e:
        print(f"   ❌ Data loading error: {e}")
        traceback.print_exc()
        return

    # 3. Verify Model
    print("\n3. Checking Model Architecture...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ConvNeXtClassifier(
            num_classes=len(config.data.classes),
            pretrained=False  # No need for weights for structure check
        ).to(device)
        
        print(f"   Model created: ConvNeXt-Large")
        print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
        print("   ✅ Model initialization successful")
        
        # 4. Verify Forward Pass & Loss
        print("\n4. Checking Forward Pass & Loss...")
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward
        outputs = model(images)
        print(f"   Outputs shape: {outputs.shape}")
        
        if outputs.shape != targets.shape:
             raise ValueError(f"Shape mismatch: {outputs.shape} vs {targets.shape}")
             
        # Loss
        alpha = torch.tensor(train_dataset.class_weights)
        pos_weight = torch.tensor(train_dataset.pos_weights)
        criterion = create_loss(
            loss_type='focal',
            alpha=alpha,
            pos_weight=pos_weight,
            gamma=2.0
        )
        
        loss = criterion(outputs, targets)
        print(f"   Loss value: {loss.item():.4f}")
        
        # Backward
        loss.backward()
        print("   ✅ Backward pass successful")
        
    except Exception as e:
        print(f"   ❌ Model error: {e}")
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE - READY FOR TRAINING")
    print("=" * 60)

if __name__ == "__main__":
    verify_pipeline()
