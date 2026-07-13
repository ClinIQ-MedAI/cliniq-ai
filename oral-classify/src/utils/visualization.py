# visualize_data.py

import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, IMAGENET_MEAN, IMAGENET_STD
from model import get_datasets

# =========================================
# 1. Load datasets (train / val / test)
# =========================================
train_ds, val_ds, test_ds = get_datasets()
class_names = train_ds.classes
num_classes = len(class_names)

print("Classes:", class_names)
print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

# =========================================
# 2. Class distribution (train/val/test)
# =========================================

def get_counts(ds):
    # ImageFolder has .targets = list of class indices
    counter = Counter(ds.targets)
    return [counter[i] for i in range(num_classes)]

train_counts = get_counts(train_ds)
val_counts   = get_counts(val_ds)
test_counts  = get_counts(test_ds)

# --- Plot grouped bar chart ---
x = np.arange(num_classes)  # class indices
width = 0.25

plt.figure(figsize=(8, 5))
plt.bar(x - width, train_counts, width, label="Train")
plt.bar(x,         val_counts,   width, label="Val")
plt.bar(x + width, test_counts,  width, label="Test")

plt.xticks(x, class_names, rotation=30, ha="right")
plt.ylabel("Number of Images")
plt.title("Class Distribution (Train / Val / Test)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

dist_path = f"{OUTPUT_DIR}/class_distribution.png"
plt.savefig(dist_path, dpi=300)
plt.close()

print("Saved class distribution plot to:", dist_path)

# =========================================
# 3. Sample images per class (from train set)
# =========================================

def denormalize(t):
    # Undo ImageNet normalization to show correct colors
    t = t.clone()
    for c in range(3):
        t[c] = t[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return t

# For each class, pick one random index from train_ds
indices_per_class = []

for class_idx in range(num_classes):
    # All indices in train_ds that belong to this class
    class_indices = [i for i, target in enumerate(train_ds.targets) if target == class_idx]
    if len(class_indices) == 0:
        print(f"Warning: no samples found in TRAIN for class: {class_names[class_idx]}")
        indices_per_class.append(None)
        continue
    idx = random.choice(class_indices)
    indices_per_class.append(idx)

# We have 6 classes → 2 rows x 3 columns grid
rows, cols = 2, 3
fig, axes = plt.subplots(rows, cols, figsize=(10, 7))
fig.suptitle("Sample Image per Class (Train Set)", fontsize=14)

for ax, class_idx in zip(axes.flatten(), range(num_classes)):
    idx = indices_per_class[class_idx]
    if idx is None:
        ax.axis("off")
        continue

    img_tensor, label = train_ds[idx]
    img_show = denormalize(img_tensor).permute(1, 2, 0).numpy()
    img_show = np.clip(img_show, 0, 1)

    ax.imshow(img_show)
    ax.axis("off")
    ax.set_title(class_names[class_idx], fontsize=11)

# If there are more subplots than classes, hide the extra axes
for extra_ax in axes.flatten()[num_classes:]:
    extra_ax.axis("off")

plt.tight_layout()
plt.subplots_adjust(top=0.88)

sample_path = f"{OUTPUT_DIR}/samples_per_class.png"
plt.savefig(sample_path, dpi=300)
plt.close()

print("Saved per-class sample grid to:", sample_path)
