# Oral X‑Ray AI Feature (YOLOv8 + ConvNeXt Refinement)

Production‑ready **Dental Panoramic X‑ray analysis** feature built for the Graduation Project.  
This module implements a **two‑stage pipeline**:

1) **Object Detection (YOLOv8)** → detects findings/restorations with bounding boxes  
2) **Crop Classification (ConvNeXt)** → reclassifies each detected crop to reduce class confusion  
3) **Production Inference** → batched refinement, JSON output, visualization, and FastAPI deployment

> ✅ Designed to run entirely on a GPU server (tested on **NVIDIA RTX A6000**).

---

## What this feature does

### Tasks
- **Detection**: predict bounding boxes + class labels (9 classes)
- **Classification refinement** (optional but recommended): re‑label each YOLO box by classifying the crop with ConvNeXt
- **Deployment**: FastAPI endpoints + CLI inference

### Classes (9)
0. Apical Periodontitis  
1. Decay  
2. Wisdom Tooth  
3. Missing Tooth  
4. Dental Filling  
5. Root Canal Filling  
6. Implant  
7. Porcelain Crown  
8. Ceramic Bridge  

---

## Results (from our training runs)

### Detection (YOLOv8x @ 1280, rect fine‑tune)
Validation (imgsz=1280) on 2688 images / 18639 instances:

- **P**: 0.879  
- **R**: 0.890  
- **mAP50**: 0.914  
- **mAP50‑95**: **0.720**

Per‑class **mAP50‑95**:

| Class | mAP50‑95 |
|---|---:|
| Apical Periodontitis | 0.450 |
| Decay | 0.544 |
| Wisdom Tooth | 0.845 |
| Missing Tooth | 0.621 |
| Dental Filling | 0.699 |
| Root Canal Filling | 0.667 |
| Implant | 0.855 |
| Porcelain Crown | 0.907 |
| Ceramic Bridge | 0.889 |

### Classification (ConvNeXt‑Large on crops)
Best validation accuracy:
- **99.32%** (epoch 9)

Macro average:
- **Precision / Recall / F1**: **0.993 / 0.993 / 0.993**

Per‑class (Precision/Recall/F1):
- Apical Periodontitis: 0.992 / 0.999 / 0.995  
- Decay: 0.989 / 0.976 / 0.983  
- Wisdom Tooth: 0.995 / 0.998 / 0.996  
- Missing Tooth: 0.996 / 0.997 / 0.996  
- Dental Filling: 0.987 / 0.988 / 0.988  
- Root Canal Filling: 0.994 / 0.991 / 0.993  
- Implant: 0.999 / 0.996 / 0.997  
- Porcelain Crown: 0.989 / 0.993 / 0.991  
- Ceramic Bridge: 0.997 / 0.995 / 0.996  

> ⚠️ Note: ConvNeXt was trained on **GT crops** (from annotation boxes). In production crops come from **YOLO predictions** (noisy boxes), so we use **confidence-based refinement** to avoid wrong overrides.

---

## Project structure

> `dataset/` and `outputs/` are **gitignored** (do not commit medical images).

```text
oral-xray/
├── api_server.py                  # FastAPI server (REST API)
├── pipeline.py                    # Core YOLO + ConvNeXt production pipeline (batched)
├── run_inference.py               # CLI inference (single/batch + visualization)
├── scripts/
│   ├── coco_to_yolo.py            # COCO → YOLO conversion
│   ├── make_crops_from_yolo.py    # Generate crop dataset from YOLO labels
│   ├── train_classifier.py        # Train ConvNeXt on crops
│   └── ...                        # (optional) training helpers / reports
├── src/
│   ├── models/                    # model builders / wrappers
│   ├── training/                  # training logic (optional split)
│   └── utils/                     # plots, reports, helpers
├── configs/                       # YAML configs for training runs
├── dataset/                       # (gitignored) raw + processed data
│   ├── raw/                       # COCO format: train2017/ val2017/ annotations/
│   └── processed/
│       ├── yolo_oral/             # YOLO format images/labels + data.yaml
│       └── crops_oral/            # crop dataset for classifier
└── outputs/                       # (gitignored) experiments, weights, logs, plots
```

---

## Setup

### Requirements
- Python 3.10+
- CUDA GPU (recommended)
- Packages:
  - `ultralytics`, `torch`, `timm`, `opencv-python`, `Pillow`, `numpy`
  - `fastapi`, `uvicorn`
  - (optional) `mlflow` for experiment tracking

### Install
From repo root:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data preparation

### Expected COCO layout (raw)
Place your dataset here:
```text
dataset/raw/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Convert COCO → YOLO
```bash
python scripts/coco_to_yolo.py \
  --coco_root dataset/raw \
  --out_root  dataset/processed/yolo_oral \
  --symlink
```

Outputs:
- `dataset/processed/yolo_oral/images/{train,val}`
- `dataset/processed/yolo_oral/labels/{train,val}`
- `dataset/processed/yolo_oral/data.yaml`
- `dataset/processed/yolo_oral/classes.json`

---

## Training

### 1) YOLO detection training
Run a YOLO training config (example):
```bash
python scripts/run_yolo.py --config configs/yolo_v8x_base_1024.yaml
```

Fine‑tune at 1280 with `rect=True` + `mosaic=0` (recommended for panoramic images):
```bash
python scripts/run_yolo.py --config configs/yolo_v8x_probe_1280_rect.yaml
```

Validation on 1280:
```bash
yolo detect val \
  model=outputs/experiments/yolo_v8x_probe_1280_rect/weights/best.pt \
  data=dataset/processed/yolo_oral/data.yaml \
  imgsz=1280 device=0
```

### 2) Build crop dataset for classification
Create crops from YOLO labels:
```bash
python scripts/make_crops_from_yolo.py \
  --yolo_root dataset/processed/yolo_oral \
  --out_root  dataset/processed/crops_oral \
  --pad 0.12
```

### 3) Train ConvNeXt classifier
Example (as used in our runs):
```bash
python scripts/train_classifier.py \
  --crops_root dataset/processed/crops_oral \
  --model convnext_large \
  --batch_size 32 \
  --epochs 60
```

Artifacts are saved under:
```text
outputs/classification/convnext_large_<timestamp>/
└── presentation/   # curves + confusion matrix + per-class report
```

---

## Production inference

### A) CLI inference (server-side)
Example:
```bash
python scripts/run_inference.py \
  --image /path/to/image.png \
  --visualize
```

Typical outputs:
- JSON with detections/refined labels
- optional visualization image (boxes + labels)

### B) Run FastAPI server
Start API (recommended inside `tmux` on a server):
```bash
tmux new -s cliniq_api
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Open docs:
- `http://127.0.0.1:8000/docs`

If you are on SSH and want to use your **local browser**, do port-forward:
```bash
ssh -L 8000:127.0.0.1:8000 <user>@<server>
```
Then open:
- `http://localhost:8000/docs`

### API endpoints
- `GET /health` → health check
- `GET /classes` → class list
- `POST /predict` → JSON results
- `POST /predict/visualize` → JSON + visualization (or image output depending on implementation)

---

## Deployment options (recommended path)

### 1) Private deployment (best while developing)
- Run `uvicorn` on `127.0.0.1`
- Access via SSH port forwarding  
✅ No firewall changes needed, safest for medical data.

### 2) Public demo (when ready)
- Use a secure tunnel (e.g., Cloudflare Tunnel) + authentication
- Or deploy behind Nginx with HTTPS + API key

> ⚠️ Medical data note: avoid storing images by default; log only metadata unless you have permission.

---

## Notes / Best practices
- **Do not commit** `dataset/` or `outputs/` to GitHub.
- Ensure **class order** stays identical across:
  - YOLO `data.yaml` names
  - crop dataset folders
  - ConvNeXt output layer
  - production pipeline mapping
- Prefer **confidence-based refinement**:
  - only override YOLO label if classifier confidence ≥ threshold
- For robustness: evaluate ConvNeXt on **YOLO crops** (predicted boxes) not just GT crops.

---

## Quick demo commands (copy/paste)

```bash
# 1) Convert dataset
python scripts/coco_to_yolo.py --coco_root dataset/raw --out_root dataset/processed/yolo_oral --symlink

# 2) Train YOLO (base)
python scripts/run_yolo.py --config configs/yolo_v8x_base_1024.yaml

# 3) Fine-tune YOLO @1280 rect
python scripts/run_yolo.py --config configs/yolo_v8x_probe_1280_rect.yaml

# 4) Make crops dataset
python scripts/make_crops_from_yolo.py --yolo_root dataset/processed/yolo_oral --out_root dataset/processed/crops_oral --pad 0.12

# 5) Train classifier
python scripts/train_classifier.py --crops_root dataset/processed/crops_oral --model convnext_large --batch_size 32 --epochs 60

# 6) Run API
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

---

## Acknowledgements
- Detection backbone: Ultralytics YOLOv8  
- Classification backbone: ConvNeXt (timm)

---

## Disclaimer
This system is for research/educational use and must be validated clinically before any medical usage.
