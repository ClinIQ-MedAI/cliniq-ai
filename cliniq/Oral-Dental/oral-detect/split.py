import shutil
import random
from pathlib import Path

# ===== CONFIG =====
DATA_DIR = Path("/home/moabouag/far/oral-detect/oral_yolo_dataset/Data")
OUT_DIR  = Path("/home/moabouag/far/oral-detect/oral_yolo_dataset/Data_split")

VAL_RATIO = 0.20
IMG_EXTS = {".jpg", ".jpeg", ".png"}

def main():
    print("Creating clean YOLO split...")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "images/train").mkdir(parents=True)
    (OUT_DIR / "images/val").mkdir(parents=True)
    (OUT_DIR / "labels/train").mkdir(parents=True)
    (OUT_DIR / "labels/val").mkdir(parents=True)

    # Collect all images in Data/images (train + val)
    all_images = list((DATA_DIR / "images").rglob("*"))
    all_images = [p for p in all_images if p.suffix.lower() in IMG_EXTS]

    # Match labels
    pairs = []
    for img in all_images:
        label = DATA_DIR / "labels" / img.relative_to(DATA_DIR / "images")
        label = label.with_suffix(".txt")
        if label.exists():
            pairs.append((img, label))

    print(f"Total valid pairs: {len(pairs)}")

    # Shuffle
    random.seed(42)
    random.shuffle(pairs)

    # Split
    val_count = int(len(pairs) * VAL_RATIO)
    val_set = pairs[:val_count]
    train_set = pairs[val_count:]

    # Copy files
    for img, lbl in train_set:
        shutil.copy(img, OUT_DIR / "images/train" / img.name)
        shutil.copy(lbl, OUT_DIR / "labels/train" / lbl.name)

    for img, lbl in val_set:
        shutil.copy(img, OUT_DIR / "images/val" / img.name)
        shutil.copy(lbl, OUT_DIR / "labels/val" / lbl.name)

    print(f"Train: {len(train_set)}")
    print(f"Val:   {len(val_set)}")
    print("DONE!")

if __name__ == "__main__":
    main()
