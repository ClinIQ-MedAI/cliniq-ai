# model.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models as tv_models
from torch.utils.data import DataLoader

from config import (
    DATA_DIR, BATCH, NUM_CLASSES,
    IMAGENET_MEAN, IMAGENET_STD,
    DEVICE
)


def get_transforms():
    """Return train and test/val transforms — optimized for ConvNeXt."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),           # helpful for oral images
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, test_tf


def get_datasets():
    """Return train, val, test datasets (ImageFolder)."""
    train_tf, test_tf = get_transforms()

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=test_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=test_tf)

    return train_ds, val_ds, test_ds


def get_dataloaders():
    """Return dataloaders and class names."""
    train_ds, val_ds, test_ds = get_datasets()

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    class_names = train_ds.classes
    print("Classes:", class_names)

    return train_dl, val_dl, test_dl, class_names, test_ds


def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.5):
    """
    Build ConvNeXt-Tiny with pretrained weights and custom head.
    ConvNeXt uses classifier[2] as the final Linear layer.
    """
    try:
        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    except AttributeError:
        weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT

    net = tv_models.convnext_tiny(weights=weights)

    # Replace the final classification head
    in_features = net.classifier[2].in_features
    net.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    # Optional: freeze early layers for first few epochs (uncomment if needed)
    # for param in net.features[:6].parameters():
    #     param.requires_grad = False

    net = net.to(DEVICE)
    return net