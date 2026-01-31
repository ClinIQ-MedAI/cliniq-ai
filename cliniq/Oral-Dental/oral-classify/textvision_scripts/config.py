# config.py
# Centralized configuration for TextVision OCR pipeline.

from dataclasses import dataclass

@dataclass
class Paths:
    # Base data root (change if not using Colab paths)
    data_root: str = "/content"

    # Raw downloaded zips (optional)
    train_zip: str = "/content/train.zip"
    test_zip: str = "/content/test.zip"

    # Unzipped raw folders
    train_dir: str = "/content/train"
    test_dir: str = "/content/test"

    # Word-level dataset output
    word_dataset_dir: str = "/content/word_dataset"
    word_images_dir: str = "/content/word_dataset/images"
    labels_file: str = "/content/word_dataset/labels.txt"
    train_list: str = "/content/word_dataset/train.txt"
    val_list: str = "/content/word_dataset/val.txt"

@dataclass
class Training:
    img_h: int = 32
    img_w: int = 128
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    num_workers: int = 2
    pin_memory: bool = True
    grad_clip: float = 5.0
    save_dir: str = "/content/checkpoints"
