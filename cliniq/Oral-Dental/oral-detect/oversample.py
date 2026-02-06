
import shutil
from pathlib import Path
import random

# =========================== CONFIG ===========================
DATA_DIR = Path("/home/moabouag/far/oral-detect/oral_yolo_dataset/Data")         
OUT_DIR  = Path("/home/moabouag/far/oral-detect/oral_yolo_dataset/Data_balanced")

CLASSES = ['Caries', 'Ulcer', 'Tooth Discoloration', 'Gingivitis']
NUM_CLASSES = len(CLASSES)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MAX_FACTOR = 3.5
random.seed(42)

# =================================================================
def find_label(stem):
    candidates = list(DATA_DIR.glob(f"labels/**/*{stem}.txt"))
    return candidates[0] if candidates else None

def collect_pairs():
    pairs = []
    counts = {i: 0 for i in range(NUM_CLASSES)}
    for img_path in DATA_DIR.glob("images/**/*"):
        if img_path.suffix.lower() not in IMG_EXTS or not img_path.is_file():
            continue
        lbl_path = find_label(img_path.stem)
        if not lbl_path:
            continue
        cls_ids = [int(l.split()[0]) for l in lbl_path.read_text().splitlines() if l.strip()]
        for cid in cls_ids:
            counts[cid] += 1
        pairs.append((img_path, lbl_path, cls_ids))
    return pairs, counts

def compute_factors(counts):
    max_c = max(counts.values())
    return {i: min(max_c/counts[i], MAX_FACTOR) if counts[i]>0 else 1.0 for i in range(NUM_CLASSES)}

def copy_with_unique_name(src, dst_dir):
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst.name
    
    stem = src.stem
    ext = src.suffix
    i = 1
    while True:
        new_name = f"{stem}_copy{i}{ext}"
        new_path = dst_dir / new_name
        if not new_path.exists():
            shutil.copy2(src, new_path)
            return new_name
        i += 1

# =================================================================
if __name__ == "__main__":
    print("Collecting original data...")
    pairs, counts = collect_pairs()
    print("Original →", {CLASSES[i]: counts[i] for i in range(NUM_CLASSES)})

    factors = compute_factors(counts)
    print("Factors  →", {CLASSES[i]: f"×{factors[i]:.2f}" for i in range(NUM_CLASSES)})

    # Oversample
    train_samples = []
    for img_path, lbl_path, cls_list in pairs:
        factor = max(factors.get(cid, 1.0) for cid in set(cls_list))
        repeats = max(1, round(factor))
        train_samples.extend([(img_path, lbl_path)] * repeats)

    print(f"Oversampled train size → {len(train_samples)}")

    # Clean output folder
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "images/train").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images/val").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels/train").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels/val").mkdir(parents=True, exist_ok=True)

    # Copy oversampled train with unique names
    for img_path, lbl_path in train_samples:
        img_name = copy_with_unique_name(img_path, OUT_DIR / "images/train")
        lbl_name = copy_with_unique_name(lbl_path, OUT_DIR / "labels/train")
        # Keep same name for label as image
        correct_lbl = OUT_DIR / "labels/train" / Path(img_name).with_suffix(".txt")
        if OUT_DIR / "labels/train" / lbl_name != correct_lbl:
            shutil.move(OUT_DIR / "labels/train" / lbl_name, correct_lbl)

    # Copy original val (unchanged)
    for img_path in (DATA_DIR / "images/val").rglob("*"):
        if img_path.suffix.lower() in IMG_EXTS:
            lbl_path = find_label(img_path.stem)
            if lbl_path:
                shutil.copy2(img_path, OUT_DIR / "images/val" / img_path.name)
                shutil.copy2(lbl_path, OUT_DIR / "labels/val" / img_path.name.replace(img_path.suffix, ".txt"))

    print(f"\nFINAL BALANCED DATASET READY")
    print(f"   Train images → {len(list((OUT_DIR/'images/train').iterdir()))}")
    print(f"   Val   images → {len(list((OUT_DIR/'images/val').iterdir()))}")
    print(f"   Location: {OUT_DIR}")