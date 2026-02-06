import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# =============================
# 0. Basic config
# =============================

LOG_EVERY = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", DEVICE)

DATA_DIR = "dataset"
BATCH = 32
NUM_CLASSES = 6
EPOCHS = 25
LR = 1e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# =============================
# 1. Data transforms
# =============================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # crop + zoom
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# =============================
# 2. Datasets & Dataloaders
# =============================
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=test_tf)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=test_tf)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

print("Classes:", train_ds.classes)

# =============================
# 3. Model: ResNet50 + new head
# =============================


weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)

in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.1, patience=3
)

# =============================
# 4. Training & Validation loops
# =============================


history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

def run_epoch_train(epoch):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    print(f"\n===== Epoch {epoch+1}/{EPOCHS} — TRAIN =====")
    for batch_idx, (imgs, labels) in enumerate(train_dl, start=1):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        if batch_idx % 10 == 0 or batch_idx == len(train_dl):
            avg_loss_so_far = epoch_loss / batch_idx
            acc_so_far = 100.0 * correct / total
            print(
                f"[TRAIN] Step {batch_idx}/{len(train_dl)} | "
                f"Batch Loss: {loss.item():.4f} | "
                f"Avg Loss: {avg_loss_so_far:.4f} | "
                f"Acc: {acc_so_far:.2f}%"
            )

    final_train_loss = epoch_loss / len(train_dl)
    final_train_acc = 100.0 * correct / total
    print(f"--> TRAIN Epoch {epoch+1}: Loss = {final_train_loss:.4f}, Acc = {final_train_acc:.2f}%")
    return final_train_loss, final_train_acc


def run_epoch_eval(epoch, dataloader, split_name="VAL"):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_loss = epoch_loss / len(dataloader)
    acc = 100.0 * correct / total
    print(f"--> {split_name} Epoch {epoch+1}: Loss = {avg_loss:.4f}, Acc = {acc:.2f}%")
    return avg_loss, acc

# =============================
# 5. Full training process
# =============================
best_val_acc = 0.0
best_model_path = "best_resnet50_oral_2.pth"

for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch_train(epoch)
    val_loss, val_acc = run_epoch_eval(epoch, val_dl, split_name="VAL")

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    scheduler.step(val_acc)
    print(f"[LR Scheduler] Current LR = {optimizer.param_groups[0]['lr']}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"*** New best model saved (val_acc = {val_acc:.2f}%) ***")

print(f"\nTraining finished. Best val acc: {best_val_acc:.2f}%")
print(f"Best model saved to: {best_model_path}")

# Optionally load best model before final test
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =============================
# 6. Final test evaluation
# =============================
all_preds = []
all_actual = []

with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_actual.extend(labels.numpy())

print("\nCLASSIFICATION REPORT:")
print(classification_report(all_actual, all_preds, target_names=train_ds.classes))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(all_actual, all_preds))


# =============================
# 7. Plotting training curves
# =============================

import csv

# Save history to CSV
with open("training_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    for i in range(EPOCHS):
        writer.writerow([
            i+1,
            history["train_loss"][i],
            history["train_acc"][i],
            history["val_loss"][i],
            history["val_acc"][i],
        ])

# Plot Loss Curve
epochs_range = range(1, EPOCHS+1)

plt.figure(figsize=(6,4))
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.close()

# Plot Accuracy Curve
plt.figure(figsize=(6,4))
plt.plot(epochs_range, history["train_acc"], label="Train Acc")
plt.plot(epochs_range, history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300)
plt.close()

print("Saved loss_curve.png and accuracy_curve.png and training_history.csv")


# =============================
# 8. Confusion Matrix on Test Set
# =============================

# Load best model for test
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

all_preds = []
all_actual = []

with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_actual.extend(labels.numpy())

print("\nCLASSIFICATION REPORT:")
print(classification_report(all_actual, all_preds, target_names=train_ds.classes))

cm = confusion_matrix(all_actual, all_preds)
print("\nCONFUSION MATRIX RAW:")
print(cm)

# Plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_ds.classes)
disp.plot(cmap="Blues", values_format="d", xticks_rotation=45)
plt.title("Confusion Matrix - Oral Disease Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("Saved confusion_matrix.png")



# =============================
# 9. Sample prediction grid (for presentation)
# =============================

import random

from matplotlib import pyplot as plt

def denormalize(t):
    t = t.clone()
    for c in range(3):
        t[c] = t[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return t

indices = random.sample(range(len(test_ds)), 9)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle("Sample Predictions (True vs Pred)", fontsize=14)

with torch.no_grad():
    for ax, idx in zip(axes.flatten(), indices):
        img_tensor, true_label = test_ds[idx]
        img_batch = img_tensor.unsqueeze(0).to(DEVICE)

        out = model(img_batch)
        pred = torch.argmax(out, dim=1).item()

        true_class = train_ds.classes[true_label]
        pred_class = train_ds.classes[pred]
        status = "✓" if pred == true_label else "✗"

        img_show = denormalize(img_tensor).permute(1, 2, 0).numpy()
        img_show = np.clip(img_show, 0, 1)

        ax.imshow(img_show)
        ax.axis("off")
        ax.set_title(f"T: {true_class}\nP: {pred_class} {status}",
                     fontsize=9,
                     color=("green" if status == "✓" else "red"))

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("sample_predictions.png", dpi=300)
plt.close()

print("Saved sample_predictions.png")

