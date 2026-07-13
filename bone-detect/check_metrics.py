"""Quick check of per-class metrics from latest checkpoint."""
from ultralytics import YOLO

CHECKPOINT = "outputs/YOLO11x_TOP3_20260203_0645/weights/last.pt"
CLASS_NAMES = ["fracture", "metal", "periostealreaction"]

print("Loading model and running validation...")
model = YOLO(CHECKPOINT)
results = model.val()

print("\n" + "═"*60)
print("PER-CLASS AP@0.5 RESULTS:")
print("═"*60)

ap50_per_class = results.box.ap50
for i, name in enumerate(CLASS_NAMES):
    if i < len(ap50_per_class):
        ap = float(ap50_per_class[i])
        bar = "█" * int(ap * 20)
        print(f"  {name:20s}: {ap:.4f} {bar}")

print(f"\n  Overall mAP@0.5: {results.box.map50:.4f}")
print(f"  Overall mAP@0.5:0.95: {results.box.map:.4f}")
print("═"*60)
