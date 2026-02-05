"""
Dataset and Data Loading for Chest X-ray Multi-Label Classification
- Patient-level train/val/test split
- CLAHE preprocessing
- Allowed augmentations only
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from typing import List, Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2

from config import config


from .transforms import get_train_transforms, get_val_transforms


class ChestXrayDataset(Dataset):
    """
    NIH Chest X-ray14 Dataset for multi-label classification.
    
    Features:
    - Excludes Hernia class
    - Keeps No Finding as negative class
    - Grayscale → 3 channels for pretrained models
    - Patient-level data splitting
    """
    
    CLASSES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Infiltration", 
        "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
    ]
    
    def __init__(
        self,
        root_dir: str,
        image_list: List[str],
        labels_df: pd.DataFrame,
        classes: Optional[List[str]] = None,
        transform: Optional[T.Compose] = None,
        return_metadata: bool = False
    ):
        self.root_dir = root_dir
        self.image_list = image_list
        self.labels_df = labels_df
        self.classes = classes or self.CLASSES
        self.num_classes = len(self.classes)
        self.transform = transform
        self.return_metadata = return_metadata
        
        # Build labels dict
        self.labels_dict = dict(zip(
            labels_df['Image Index'],
            labels_df['Finding Labels']
        ))
        
        # Build patient ID dict
        self.patient_dict = dict(zip(
            labels_df['Image Index'],
            labels_df['Patient ID']
        ))
        
        # Class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build image path lookup
        self.image_paths = self._build_image_paths()
        
        # Compute class weights
        self.class_weights = self._compute_class_weights()
        
        # Compute positive weights for BCE
        self.pos_weights = self._compute_pos_weights()
    
    def _build_image_paths(self) -> Dict[str, str]:
        """Build image path lookup from all subdirectories."""
        paths = {}
        for i in range(1, 13):
            dir_path = os.path.join(self.root_dir, f"images_{i:03d}", "images")
            if os.path.exists(dir_path):
                for img_name in os.listdir(dir_path):
                    paths[img_name] = os.path.join(dir_path, img_name)
        return paths
    
    def _compute_class_weights(self) -> np.ndarray:
        """Compute inverse frequency class weights."""
        counts = np.zeros(self.num_classes)
        
        for img_name in self.image_list:
            if img_name in self.labels_dict:
                labels_str = self.labels_dict[img_name]
                for label in labels_str.split('|'):
                    label = label.strip()
                    if label in self.class_to_idx:
                        counts[self.class_to_idx[label]] += 1
        
        # Inverse frequency
        total = len(self.image_list)
        weights = total / (counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        
        return weights.astype(np.float32)
    
    def _compute_pos_weights(self) -> np.ndarray:
        """Compute positive weights (neg_count / pos_count) for BCE."""
        pos_counts = np.zeros(self.num_classes)
        
        for img_name in self.image_list:
            if img_name in self.labels_dict:
                labels_str = self.labels_dict[img_name]
                for label in labels_str.split('|'):
                    label = label.strip()
                    if label in self.class_to_idx:
                        pos_counts[self.class_to_idx[label]] += 1
        
        neg_counts = len(self.image_list) - pos_counts
        pos_weights = neg_counts / (pos_counts + 1e-6)
        
        return pos_weights.astype(np.float32)
    
    def _encode_labels(self, label_string: str) -> np.ndarray:
        """Convert label string to multi-hot vector."""
        labels = np.zeros(self.num_classes, dtype=np.float32)
        
        for label in label_string.split('|'):
            label = label.strip()
            if label in self.class_to_idx:
                labels[self.class_to_idx[label]] = 1.0
        
        return labels
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, idx: int):
        img_name = self.image_list[idx]
        
        # Load image
        img_path = self.image_paths.get(img_name)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
        
        # Load as grayscale and convert to RGB (3 channels)
        image = Image.open(img_path).convert('L')  # Grayscale
        image = image.convert('RGB')  # Repeat to 3 channels
        
        # Get labels
        label_string = self.labels_dict.get(img_name, "No Finding")
        labels = self._encode_labels(label_string)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.return_metadata:
            return image, labels, img_path
        
        return image, labels


def create_patient_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split data at patient level to prevent data leakage.
    
    Args:
        df: DataFrame with 'Image Index' and 'Patient ID'
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed
    
    Returns:
        (train_images, val_images, test_images)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Group images by patient
    patient_to_images = defaultdict(list)
    for _, row in df.iterrows():
        patient_to_images[row['Patient ID']].append(row['Image Index'])
    
    # Get all unique patients
    patients = list(patient_to_images.keys())
    random.shuffle(patients)
    
    # Split patients
    n_patients = len(patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    
    # Get images for each split
    train_images = [img for p in train_patients for img in patient_to_images[p]]
    val_images = [img for p in val_patients for img in patient_to_images[p]]
    test_images = [img for p in test_patients for img in patient_to_images[p]]
    
    print(f"Patient-level split:")
    print(f"  Train: {len(train_patients):,} patients, {len(train_images):,} images")
    print(f"  Val: {len(val_patients):,} patients, {len(val_images):,} images")
    if test_patients:
        print(f"  Unused/Test: {len(test_patients):,} patients, {len(test_images):,} images")
    
    return train_images, val_images, test_images


def get_dataloaders(
    batch_size: int = 4,
    image_size: int = 512,
    num_workers: int = 8,
    use_official_split: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, ChestXrayDataset]:
    """
    Create train, val, test dataloaders.
    
    Args:
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers
        use_official_split: Use official NIH train/test split
        seed: Random seed
    
    Returns:
        (train_loader, val_loader, test_loader, train_dataset)
    """
    cfg = config.data
    
    # Load labels
    labels_df = pd.read_csv(os.path.join(cfg.root_dir, cfg.labels_file))
    
    # Filter out Hernia
    labels_df = labels_df[~labels_df['Finding Labels'].str.contains('Hernia')]
    
    if use_official_split:
        # Load official splits
        with open(os.path.join(cfg.root_dir, cfg.train_list), 'r') as f:
            train_val_images = set(f.read().strip().split('\n'))
        
        with open(os.path.join(cfg.root_dir, cfg.test_list), 'r') as f:
            test_images = list(f.read().strip().split('\n'))
        
        # Filter to valid images (excluding Hernia)
        valid_images = set(labels_df['Image Index'])
        train_val_images = [img for img in train_val_images if img in valid_images]
        test_images = [img for img in test_images if img in valid_images]
        
        # Split train_val into train and val (patient-level)
        train_val_df = labels_df[labels_df['Image Index'].isin(train_val_images)]
        train_images, val_images, _ = create_patient_level_split(
            train_val_df, train_ratio=0.85, val_ratio=0.15, seed=seed
        )
    else:
        # Full patient-level split
        train_images, val_images, test_images = create_patient_level_split(
            labels_df, train_ratio=0.7, val_ratio=0.15, seed=seed
        )
    
    # Create datasets
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    train_dataset = ChestXrayDataset(
        root_dir=cfg.root_dir,
        image_list=train_images,
        labels_df=labels_df,
        classes=cfg.classes,
        transform=train_transform
    )
    
    val_dataset = ChestXrayDataset(
        root_dir=cfg.root_dir,
        image_list=val_images,
        labels_df=labels_df,
        classes=cfg.classes,
        transform=val_transform
    )
    
    test_dataset = ChestXrayDataset(
        root_dir=cfg.root_dir,
        image_list=test_images,
        labels_df=labels_df,
        classes=cfg.classes,
        transform=val_transform,
        return_metadata=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_dataset):,} images, {len(train_loader):,} batches")
    print(f"  Val: {len(val_dataset):,} images, {len(val_loader):,} batches")
    print(f"  Test: {len(test_dataset):,} images, {len(test_loader):,} batches")
    
    return train_loader, val_loader, test_loader, train_dataset


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(
        batch_size=4,
        image_size=512,
        num_workers=4
    )
    
    # Test a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"\nClass weights: {train_dataset.class_weights}")
    print(f"\nPos weights: {train_dataset.pos_weights}")
