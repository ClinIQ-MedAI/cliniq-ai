# utils.py
# Utility helpers used across scripts.

from __future__ import annotations
import os
import random
from typing import Dict, List, Tuple
from PIL import Image

def seed_everything(seed: int = 42) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def polygon_to_bbox(poly: Dict[str, int]) -> Tuple[int, int, int, int]:
    """Convert 4-point polygon dict to minimal axis-aligned bbox."""
    xs = [poly["x0"], poly["x1"], poly["x2"], poly["x3"]]
    ys = [poly["y0"], poly["y1"], poly["y2"], poly["y3"]]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    return x_min, y_min, x_max, y_max

def crop_word(img: Image.Image, bbox: Tuple[int,int,int,int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    # Clamp to image bounds to avoid exceptions
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.width, x2); y2 = min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        # Return a 1x1 placeholder (will be filtered later if needed)
        return Image.new("RGB", (1,1), (0,0,0))
    return img.crop((x1, y1, x2, y2))

def list_images(directory: str, exts=(".png",".jpg",".jpeg")) -> List[str]:
    out=[]
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(root,f))
    return out
