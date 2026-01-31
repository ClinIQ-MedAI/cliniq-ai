# dataset.py
# Dataset for word-level OCR training from train.txt/val.txt lists.

from __future__ import annotations
import os
from typing import Tuple, List
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class OCRWordDataset(Dataset):
    def __init__(self, list_path: str, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(list_path, "r", encoding="utf-8") as f:
            self.lines = [l.strip().split("\t", maxsplit=2) for l in f if l.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", str]:
        filename, text = self.lines[idx][0], self.lines[idx][1]
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, text

def default_transform(img_h: int = 32, img_w: int = 128):
    return T.Compose([
        T.Resize((img_h, img_w)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
