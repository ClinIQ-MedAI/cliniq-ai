import os
import shutil
from pathlib import Path
import splitfolders

RAW_DATA = "/home/moabouag/.cache/kagglehub/datasets/salmansajid05/oral-diseases/versions/3"
OUT_DATA = "perfect_dataset_clean"
FINAL_DATASET = "dataset_clean"

CLASSES = ["Caries", "Gingivitis", "Discoloration", "Ulcer", "Calculus", "Hypodontia"]

# Create output folders
os.makedirs(OUT_DATA, exist_ok=True)
for c in CLASSES:
    os.makedirs(os.path.join(OUT_DATA, c), exist_ok=True)

print("BUILDING CLEAN ORAL DATASET (NO YOLO)...")

# ==============================
# 1) Function to copy images
# ==============================
def copy_from_folder(folder_name, target_class):
    folder_path = os.path.join(RAW_DATA, folder_name)
    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder not found: {folder_path}")
        return 0

    count = 0
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(root, f)
                dst = os.path.join(OUT_DATA, target_class, f"clean_{count:05d}_{f}")
                shutil.copy2(src, dst)
                count += 1

    print(f"{folder_name:30} → {target_class:12}: {count} images")
    return count

# ==============================
# 2) Copy each clean class
# ==============================
copy_from_folder("Calculus", "Calculus")
copy_from_folder("Gingivitis", "Gingivitis")
copy_from_folder("Mouth Ulcer", "Ulcer")
copy_from_folder("Tooth Discoloration", "Discoloration")
copy_from_folder("hypodontia", "Hypodontia")

# Caries dataset (nested folder)
caries_path = os.path.join(RAW_DATA, "Data caries")
caries_count = 0
for root, _, files in os.walk(caries_path):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            src = os.path.join(root, f)
            dst = os.path.join(OUT_DATA, "Caries", f"clean_caries_{caries_count:05d}_{f}")
            shutil.copy2(src, dst)
            caries_count += 1
print(f"{'Data caries':30} → {'Caries':12}: {caries_count} images")

# ==============================
# 3) Final summary
# ==============================
print("\n" + "="*70)
print("CLEAN ORAL DATASET — SUMMARY")
total = 0
for cls in CLASSES:
    num = len([
        f for f in os.listdir(os.path.join(OUT_DATA, cls))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"{cls:15}: {num:5d} images")
    total += num
print(f"TOTAL IMAGES: {total}")
print("="*70)

# ==============================
# 4) Train/Val/Test Split
# ==============================
print("\nSplitting into train/val/test...")

splitfolders.ratio(
    OUT_DATA,
    output=FINAL_DATASET,
    seed=42,
    ratio=(0.8, 0.1, 0.1)
)

print(f"\nDATASET READY → {FINAL_DATASET}/train | val | test")
