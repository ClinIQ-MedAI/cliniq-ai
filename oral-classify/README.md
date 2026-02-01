# Oral Disease Classification (ConvNeXt + GradCAM)

AI‑powered **oral disease classification** using **ConvNeXt‑Small** with **GradCAM/GradCAM++ visualization** for interpretability.

This feature provides:
- ✅ **Training + evaluation** scripts for image classification
- ✅ **FastAPI inference server** (Swagger UI at `/docs`)
- ✅ **GradCAM heatmap endpoint** to visualize model attention
- ✅ Clean modular structure (`api/`, `src/`, `scripts/`, `configs/`)

---

## Model performance

> Reported run (ConvNeXt‑Small, 6 classes)

| Metric | Value |
|---|---:|
| **Accuracy** | **94.83%** |
| **Architecture** | ConvNeXt‑Small |
| **Classes** | 6 |

### Per‑class metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Calculus | 0.81 | 0.84 | 0.83 |
| Caries | 0.99 | 0.98 | 0.99 |
| Discoloration | 0.99 | 0.99 | 0.99 |
| Gingivitis | 0.86 | 0.93 | 0.89 |
| Hypodontia | 1.00 | 0.99 | 0.99 |
| Ulcer | 1.00 | 1.00 | 1.00 |

---

## Classes

1. **Calculus** — Tartar buildup  
2. **Caries** — Tooth decay / cavities  
3. **Discoloration** — Tooth staining  
4. **Gingivitis** — Gum inflammation  
5. **Hypodontia** — Missing teeth  
6. **Ulcer** — Oral sores  

---

## Project structure

```text
oral-classify/
├── api/                    # FastAPI server + GradCAM
│   ├── server.py          # API endpoints
│   ├── inference.py       # Model loading & prediction
│   └── gradcam.py         # GradCAM/GradCAM++ visualization
├── src/
│   ├── models/            # Model architectures (ConvNeXt, heads, etc.)
│   ├── data/              # Dataset utilities / transforms / loaders
│   └── utils/             # Metrics, plots, helpers
├── scripts/               # Training & evaluation scripts
├── configs/               # Configuration files (YAML/JSON)
└── outputs/               # Trained weights + reports (NOT committed to git)
    └── weights/           # Suggested location for best_model.pth
```

---

## Quick start

### 1) Installation

You can use either **conda** or **venv**.

#### Option A — conda
```bash
conda activate cliniq
pip install -r requirements.txt
```

---

### 2) Run the inference API

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8001
```

Open:
- `http://localhost:8001/docs` (Swagger UI)
- `http://localhost:8001/health` (health check)

> If you run on a remote GPU server (SSH), use port‑forwarding:
```bash
ssh -L 8001:127.0.0.1:8001 <user>@<server>
```
Then open locally:
- `http://localhost:8001/docs`

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/classes` | GET | List disease classes |
| `/predict` | POST | Classify an image (JSON) |
| `/predict/gradcam` | POST | Classify + GradCAM heatmap overlay |

### Example (Python)

```python
import requests

# Predict
with open("dental_image.jpg", "rb") as f:
    r = requests.post("http://localhost:8001/predict", files={"file": f})
    print(r.json())

# GradCAM visualization (image response)
with open("dental_image.jpg", "rb") as f:
    r = requests.post("http://localhost:8001/predict/gradcam", files={"file": f})
    with open("gradcam_result.jpg", "wb") as out:
        out.write(r.content)
```

### Example (curl)

```bash
# Predict
curl -X POST "http://localhost:8001/predict" \
  -F "file=@dental_image.jpg"

# GradCAM overlay saved to file
curl -X POST "http://localhost:8001/predict/gradcam" \
  -F "file=@dental_image.jpg" \
  --output gradcam_result.jpg
```

---

## Weights

Model weights are **not included** in the repository.

### Option A — Train your own weights
```bash
cd scripts
python train_convnext.py
```

### Option B — Use a pre‑trained checkpoint
Place your weights here (recommended convention):
```text
outputs/weights/best_model.pth
```

If your code expects a different filename/path, update it inside:
- `api/inference.py` (model loading path)

> Tip: in production, keep a `weights/` folder and a single `config.yaml` that pins:
> - model architecture
> - class list order
> - checkpoint path
> - preprocessing settings

---

## Training & evaluation

### Train
```bash
python scripts/train_convnext.py
```

### What training should output
Typical artifacts you’ll want under `outputs/`:
- `best_model.pth`
- `metrics.json` (accuracy, macro‑F1, per‑class report)
- `confusion_matrix.png`
- `loss_curve.png`, `acc_curve.png` (optional)

---

## GradCAM visualization

GradCAM (Gradient‑weighted Class Activation Mapping) highlights which regions of the image contributed most to the model’s prediction.

Why it matters:
- **Interpretability**: understand model decisions
- **Validation**: ensure attention is on medically relevant areas
- **Trust**: more confidence for demo/clinical review

---

## Production deployment guide

### 1) Run as a background service (recommended)
Use `tmux` so SSH disconnect won’t stop the server:

```bash
tmux new -s oral_classify_api
uvicorn api.server:app --host 0.0.0.0 --port 8001
# detach: Ctrl+b then d
```

### 2) Reverse proxy (optional)
If you later put it behind Nginx:
- run uvicorn on `127.0.0.1:8001`
- terminate SSL at Nginx
- add request size limits for image upload

### 3) Basic safety controls (recommended)
- Add API key auth (header‑based) for public URLs
- Disable saving uploaded images by default
- Log only metadata (timings, predicted class), not raw images

---

## Troubleshooting

### Swagger doesn’t show “Choose File”
Your FastAPI endpoint must accept:
```python
file: UploadFile = File(...)
```

### CUDA OOM
- Reduce batch size during training
- Use smaller input size (e.g., 224 is standard for ConvNeXt)
- Ensure only one uvicorn worker loads the GPU model

---

## Disclaimer
This project is for research/educational use. It is **not** a medical device and must be clinically validated before any real medical usage.
