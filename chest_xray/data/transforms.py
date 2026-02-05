"""
Data augmentation transforms for Chest X-ray images.
Includes custom implementations for CLAHE and Gaussian Noise.
"""

import random
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from typing import Tuple


class CLAHETransform:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Works on PIL Images, applied before ToTensor.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8), p: float = 0.5):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        
        # Convert PIL to numpy
        img_array = np.array(img)
        
        # Apply CLAHE (Handling Grayscale vs RGB)
        if len(img_array.shape) == 3:
            # Convert RGB -> LAB -> Apply to L -> RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            img_array[:, :, 0] = self.clahe.apply(img_array[:, :, 0])
            img_array = cv2.cvtColor(img_array, cv2.COLOR_LAB2RGB)
        else:
            img_array = self.clahe.apply(img_array)
        
        return Image.fromarray(img_array)


class GaussianNoise:
    """
    Add Gaussian noise to tensor.
    Applied after ToTensor.
    """
    def __init__(self, mean: float = 0, std: float = 0.02, p: float = 0.2):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0, 1)


def get_train_transforms(image_size: int = 512):
    """
    Training transforms with full augmentation pipeline.
    """
    return transforms.Compose([
        # 1. Resize first
        transforms.Resize((image_size, image_size)),
        
        # 2. CLAHE (Preprocessing Enhancement)
        CLAHETransform(clip_limit=2.0, p=0.5),
        
        # 3. Geometric Augmentations
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        
        # 4. Color Augmentations
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.15, contrast=0.15) # Adjusted to be safer
        ], p=0.3),
        
        # 5. Convert to Tensor
        transforms.ToTensor(),
        
        # 6. Gaussian Noise (Must be after ToTensor)
        GaussianNoise(mean=0, std=0.02, p=0.2),
        
        # 7. Normalization (Last step)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(image_size: int = 512):
    """
    Validation/Test transforms (Deterministic).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
