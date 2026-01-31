"""
Data augmentation and preprocessing transforms using Albumentations.

Provides:
- Training transforms with augmentation
- Validation/test transforms (resize + normalize only)
- Task-specific transforms (classification, detection, segmentation)
"""

from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def get_classification_transforms(
    image_size: int = 224,
    mode: str = 'train',
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augmentation_level: str = 'medium'
) -> A.Compose:
    """
    Get transforms for classification task.
    
    Args:
        image_size: Target image size
        mode: 'train' or 'val'
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)
        augmentation_level: 'light', 'medium', or 'heavy'
        
    Returns:
        Albumentations Compose object
    """
    if mode == 'train':
        if augmentation_level == 'light':
            transforms = [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]
        elif augmentation_level == 'medium':
            transforms = [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=10
                    ),
                    A.RandomGamma(),
                ], p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]
        else:  # heavy
            transforms = [
                A.Resize(int(image_size * 1.2), int(image_size * 1.2)),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=30,
                    p=0.7
                ),
                A.OneOf([
                    A.ElasticTransform(),
                    A.GridDistortion(),
                    A.OpticalDistortion(),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                    A.MedianBlur(blur_limit=5),
                ], p=0.4),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20
                    ),
                    A.RandomGamma(),
                    A.CLAHE(),
                ], p=0.6),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=image_size // 8,
                    max_width=image_size // 8,
                    p=0.3
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]
    else:  # val/test
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    
    return A.Compose(transforms)


def get_detection_transforms(
    image_size: int = 512,
    mode: str = 'train',
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augmentation_level: str = 'medium'
) -> A.Compose:
    """
    Get transforms for object detection task.
    
    Args:
        image_size: Target image size
        mode: 'train' or 'val'
        mean: Normalization mean
        std: Normalization std
        augmentation_level: 'light', 'medium', or 'heavy'
        
    Returns:
        Albumentations Compose object with bbox_params
    """
    bbox_params = A.BboxParams(
        format='pascal_voc',  # [x1, y1, x2, y2]
        label_fields=['labels'],
        min_visibility=0.3,
        min_area=100
    )
    
    if mode == 'train':
        if augmentation_level == 'light':
            transforms = [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]
        elif augmentation_level == 'medium':
            transforms = [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 30.0)),
                    A.GaussianBlur(blur_limit=3),
                    A.MotionBlur(blur_limit=3),
                ], p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]
        else:  # heavy
            transforms = [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.7
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=5),
                    A.MotionBlur(blur_limit=5),
                ], p=0.4),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15,
                        sat_shift_limit=25,
                        val_shift_limit=15
                    ),
                    A.CLAHE(),
                ], p=0.6),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]
    else:  # val/test
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    
    return A.Compose(transforms, bbox_params=bbox_params)


def get_segmentation_transforms(
    image_size: int = 512,
    mode: str = 'train',
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augmentation_level: str = 'medium'
) -> A.Compose:
    """
    Get transforms for segmentation task.
    
    Args:
        image_size: Target image size
        mode: 'train' or 'val'
        mean: Normalization mean
        std: Normalization std
        augmentation_level: 'light', 'medium', or 'heavy'
        
    Returns:
        Albumentations Compose object
    """
    if mode == 'train':
        if augmentation_level == 'light':
            transforms = [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]
        elif augmentation_level == 'medium':
            transforms = [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5),
                    A.GridDistortion(),
                ], p=0.3),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]
        else:  # heavy
            transforms = [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.7
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=100, sigma=10, alpha_affine=10),
                    A.GridDistortion(num_steps=5, distort_limit=0.3),
                    A.OpticalDistortion(),
                ], p=0.4),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=5),
                ], p=0.4),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15,
                        sat_shift_limit=25,
                        val_shift_limit=15
                    ),
                    A.CLAHE(),
                ], p=0.6),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]
    else:  # val/test
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    
    return A.Compose(transforms)


def get_transforms(
    task: str = 'classification',
    image_size: int = 224,
    mode: str = 'train',
    **kwargs
) -> A.Compose:
    """
    Get transforms for a specific task.
    
    Args:
        task: 'classification', 'detection', or 'segmentation'
        image_size: Target image size
        mode: 'train' or 'val'
        **kwargs: Additional arguments passed to specific transform functions
        
    Returns:
        Albumentations Compose object
    """
    if task == 'classification':
        return get_classification_transforms(image_size=image_size, mode=mode, **kwargs)
    elif task == 'detection':
        return get_detection_transforms(image_size=image_size, mode=mode, **kwargs)
    elif task == 'segmentation':
        return get_segmentation_transforms(image_size=image_size, mode=mode, **kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


# X-ray specific transforms
def get_xray_specific_transforms(
    image_size: int = 512,
    mode: str = 'train'
) -> A.Compose:
    """
    X-ray specific preprocessing and augmentation.
    
    Medical imaging specific transforms:
    - CLAHE for contrast enhancement
    - Conservative augmentation to preserve anatomical features
    - No color jittering (grayscale medical images)
    """
    if mode == 'train':
        transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            # CLAHE for X-ray contrast enhancement
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.7),
            # Conservative noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 15.0)),
                A.GaussianBlur(blur_limit=3),
            ], p=0.2),
            # Normalize
            A.Normalize(mean=(0.5,), std=(0.5,)),  # Grayscale normalization
            ToTensorV2()
        ]
    else:
        transforms = [
            A.Resize(image_size, image_size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ]
    
    return A.Compose(transforms)
