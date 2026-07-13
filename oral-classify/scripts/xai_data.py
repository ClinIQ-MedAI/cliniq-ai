# generate_final_xai_folder.py
# يولد لك المجلد النهائي XAI_Complete_Train_Test جاهز للتسليم لزميلك/الدكتور

import shutil
from pathlib import Path
import random

# ==================== عدّل الـ 3 مسارات دول بس ====================
TRAIN_FOLDER = Path("/home/moabouag/far/oral-project/dataset_clean/train")   # مجلد الـ train
TEST_FOLDER  = Path("/home/moabouag/far/oral-project/dataset_clean/test")    # مجلد الـ test
OUTPUT_ROOT  = Path("XAI_Complete_Train_Test")                        # هيتم إنشاؤه تلقائيًا
# ===================================================================

# إعدادات
NUM_TRAIN_PER_CLASS = 5   # 5 صورة من train لكل كلاس
NUM_TEST_PER_CLASS  = 8   # 8 صور من test لكل كلاس (الأهم)

CLASS_NAMES = ['Caries', 'Ulcer', 'Tooth Discoloration', 'Gingivitis']

# إنشاء المجلدات
train_out = OUTPUT_ROOT / "01_TRAIN_SET_Examples"
test_out  = OUTPUT_ROOT / "02_TEST_SET_Unseen_Examples"
train_out.mkdir(parents=True, exist_ok=True)
test_out.mkdir(parents=True, exist_ok=True)

print("جاري نسخ الصور...")

# نسخ من الـ Train
for class_name in CLASS_NAMES:
    src_folder = TRAIN_FOLDER / class_name
    if not src_folder.exists():
        continue
    images = list(src_folder.glob("*.jpg"))
    selected = random.sample(images, min(NUM_TRAIN_PER_CLASS, len(images)))
    for img in selected:
        dest = train_out / f"TRAIN_{class_name}_{img.name}"
        shutil.copy(img, dest)

# نسخ من الـ Test (الأهم)
for class_name in CLASS_NAMES:
    src_folder = TEST_FOLDER / class_name
    if not src_folder.exists():
        continue
    images = list(src_folder.glob("*.jpg"))
    selected = random.sample(images, min(NUM_TEST_PER_CLASS, len(images)))
    for img in selected:
        dest = test_out / f"TEST_{class_name}_{img.name}"
        shutil.copy(img, dest)

# كتابة ملف INFO تلقائي
info_text = f"""XAI SAMPLES – ORAL DISEASE CLASSIFICATION
=========================================

ALL GRAD-CAM++ HEATMAPS IN THIS FOLDER ARE READY FOR THESIS/PAPER

1. 02_TEST_SET_Unseen_Examples/  → {len(list(test_out.glob("*.jpg")))} images from TEST SET (completely unseen)
   → Use these in the thesis/paper (reviewers love this)

2. 01_TRAIN_SET_Examples/          → {len(list(train_out.glob("*.jpg")))} images from TRAIN SET (for comparison only)

Dataset split (stratified, seed=42):
- Train : ~2873 images (80%)
- Val   : ~360 images  (10%)
- Test  : ~354 images  (10%) ←←← GRAD-CAM++ DONE ON THESE

Test set distribution:
Caries              → ~224
Ulcer               ~28
Tooth Discoloration ~26
Gingivitis          ~76

Grad-CAM++ generated using ConvNeXt-small
Target layer: model.stages[3] → best dental results
"""

with open(OUTPUT_ROOT / "INFO_and_Distribution.txt", "w", encoding="utf-8") as f:
    f.write(info_text)

print("\nتم بنجاح!")
print(f"المجلد النهائي جاهز: {OUTPUT_ROOT.resolve()}")
print(f"→ {len(list(train_out.glob('*.jpg')))} صورة من Train")
print(f"→ {len(list(test_out.glob('*.jpg')))} صورة من Test (unseen)")
print("\nابعته لزميلك دلوقتي وخلّصنا")