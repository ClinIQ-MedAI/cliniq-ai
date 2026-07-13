import os
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

# ==============================
# Config
# ==============================
best_model_path = "best_resnet50_oral_2.pth"   # your saved checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "dataset"
NUM_CLASSES = 6
OUTPUT_DIR = "output_metrics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ==============================
# Data (test loader)
# ==============================
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
class_names = test_ds.classes
print("Classes:", class_names)

# ==============================
# Model (must match training head!)
# ==============================
weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)

in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, NUM_CLASSES)
)

model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(f"Loaded model from {best_model_path} on {DEVICE}")

# ==============================
# Collect probabilities + labels
# ==============================
all_probs = []
all_actual = []

with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)                       # logits
        probs = torch.softmax(outputs, dim=1)       # [batch, NUM_CLASSES]

        all_probs.append(probs.cpu().numpy())
        all_actual.extend(labels.numpy())

all_probs = np.concatenate(all_probs, axis=0)       # [N, NUM_CLASSES]
all_actual = np.array(all_actual)                   # [N]

# ==============================
# Compute PR curves
# ==============================
y_true_bin = label_binarize(all_actual, classes=list(range(NUM_CLASSES)))

plt.figure(figsize=(8, 6))

for i, class_name in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
    plt.plot(recall, precision, label=class_name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Per Class)")
plt.legend()
plt.grid(True)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "pr_curves.png")
plt.savefig(out_path, dpi=300)
plt.close()

print("Saved pr_curves.png to:", out_path)
