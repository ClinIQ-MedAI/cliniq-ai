"""
PyTorch Dataset classes for dental X-ray data.

Supports:
- Classification (image-level labels)
- Object Detection (bounding boxes)
- Instance Segmentation (masks)
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCOClassificationDataset(Dataset):
    """
    COCO dataset for multi-class or multi-label classification.
    
    Converts COCO detection/segmentation annotations to image-level labels.
    Each image is labeled with all categories present in its annotations.
    """
    
    def __init__(
        self,
        annotation_file: str,
        img_dir: str,
        transform: Optional[Callable] = None,
        multi_label: bool = True,
        min_area: float = 0.0
    ):
        """
        Args:
            annotation_file: Path to COCO JSON file
            img_dir: Directory containing images
            transform: Albumentation transforms
            multi_label: If True, use multi-label (multiple classes per image)
                        If False, use single label (dominant class)
            min_area: Minimum annotation area to consider for labeling
        """
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.multi_label = multi_label
        self.min_area = min_area
        
        # Load COCO
        self.coco = COCO(annotation_file)
        
        # Get all image IDs
        self.image_ids = list(self.coco.imgs.keys())
        
        # Build category mapping
        self.categories = {cat['id']: cat['name'] for cat in self.coco.dataset['categories']}
        self.num_classes = len(self.categories)
        self.cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(sorted(self.categories.keys()))}
        
        print(f"Loaded {len(self.image_ids)} images with {self.num_classes} classes")
        print(f"Categories: {list(self.categories.values())}")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        
        # Filter by area
        anns = [ann for ann in anns if ann.get('area', 0) >= self.min_area]
        
        # Get category labels
        cat_ids = [ann['category_id'] for ann in anns]
        
        if self.multi_label:
            # Multi-label: binary vector
            label = torch.zeros(self.num_classes, dtype=torch.float32)
            for cat_id in cat_ids:
                label_idx = self.cat_id_to_label[cat_id]
                label[label_idx] = 1.0
        else:
            # Single label: most frequent category
            if cat_ids:
                most_common_cat = max(set(cat_ids), key=cat_ids.count)
                label = self.cat_id_to_label[most_common_cat]
            else:
                label = 0  # Background or default class
            label = torch.tensor(label, dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'label': label,
            'image_id': img_id,
            'file_name': img_info['file_name']
        }


class COCODetectionDataset(Dataset):
    """
    COCO dataset for object detection.
    
    Returns images with bounding boxes and category labels.
    Compatible with torchvision detection models.
    """
    
    def __init__(
        self,
        annotation_file: str,
        img_dir: str,
        transform: Optional[Callable] = None,
        min_area: float = 10.0,
        remove_crowd: bool = True
    ):
        """
        Args:
            annotation_file: Path to COCO JSON file
            img_dir: Directory containing images
            transform: Albumentation transforms with bbox_params
            min_area: Minimum annotation area to keep
            remove_crowd: Remove crowd annotations
        """
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.min_area = min_area
        self.remove_crowd = remove_crowd
        
        # Load COCO
        self.coco = COCO(annotation_file)
        
        # Filter images without annotations
        self.image_ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            if len(ann_ids) > 0:
                self.image_ids.append(img_id)
        
        # Build category mapping
        self.categories = {cat['id']: cat['name'] for cat in self.coco.dataset['categories']}
        self.num_classes = len(self.categories)
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(sorted(self.categories.keys()))}  # +1 for background
        
        print(f"Loaded {len(self.image_ids)} images with {self.num_classes} classes")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        
        # Filter annotations
        filtered_anns = []
        for ann in anns:
            # Remove crowd
            if self.remove_crowd and ann.get('iscrowd', 0):
                continue
            # Check area
            if ann.get('area', 0) < self.min_area:
                continue
            filtered_anns.append(ann)
        
        # Extract boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in filtered_anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann['category_id']])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            # Albumentations format
            transformed = self.transform(
                image=image,
                bboxes=boxes.numpy() if len(boxes) > 0 else [],
                labels=labels.numpy() if len(labels) > 0 else []
            )
            image = transformed['image']
            
            if transformed['bboxes']:
                boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        return image, target


class COCOSegmentationDataset(Dataset):
    """
    COCO dataset for instance segmentation.
    
    Returns images with masks, bounding boxes, and category labels.
    Compatible with torchvision segmentation models like Mask R-CNN.
    """
    
    def __init__(
        self,
        annotation_file: str,
        img_dir: str,
        transform: Optional[Callable] = None,
        min_area: float = 10.0,
        remove_crowd: bool = True
    ):
        """
        Args:
            annotation_file: Path to COCO JSON file
            img_dir: Directory containing images
            transform: Albumentation transforms
            min_area: Minimum annotation area to keep
            remove_crowd: Remove crowd annotations
        """
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.min_area = min_area
        self.remove_crowd = remove_crowd
        
        # Load COCO
        self.coco = COCO(annotation_file)
        
        # Filter images without annotations
        self.image_ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ann_ids)
            # Keep only images with segmentation masks
            if any('segmentation' in ann for ann in anns):
                self.image_ids.append(img_id)
        
        # Build category mapping
        self.categories = {cat['id']: cat['name'] for cat in self.coco.dataset['categories']}
        self.num_classes = len(self.categories)
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(sorted(self.categories.keys()))}
        
        print(f"Loaded {len(self.image_ids)} images with segmentation masks")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        
        # Filter annotations
        filtered_anns = []
        for ann in anns:
            if self.remove_crowd and ann.get('iscrowd', 0):
                continue
            if ann.get('area', 0) < self.min_area:
                continue
            if 'segmentation' not in ann:
                continue
            filtered_anns.append(ann)
        
        # Extract masks, boxes, and labels
        masks = []
        boxes = []
        labels = []
        
        for ann in filtered_anns:
            # Get mask
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            
            # Get bbox
            x, y, w_box, h_box = ann['bbox']
            boxes.append([x, y, x + w_box, y + h_box])
            
            # Get label
            labels.append(self.cat_id_to_label[ann['category_id']])
        
        # Convert to tensors
        if masks:
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, masks=masks.numpy() if len(masks) > 0 else [])
            image = transformed['image']
            if transformed.get('masks'):
                masks = torch.as_tensor(transformed['masks'], dtype=torch.uint8)
        
        target = {
            'masks': masks,
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        return image, target


def collate_fn_detection(batch):
    """
    Custom collate function for detection dataset.
    Handles variable number of objects per image.
    """
    return tuple(zip(*batch))


def collate_fn_segmentation(batch):
    """
    Custom collate function for segmentation dataset.
    Handles variable number of objects per image.
    """
    return tuple(zip(*batch))
