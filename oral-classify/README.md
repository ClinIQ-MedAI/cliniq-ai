# Oral Disease Classification

AI-powered oral disease classification using ConvNeXt-Small with GradCAM visualization.

## Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.83% |
| **Architecture** | ConvNeXt-Small |
| **Classes** | 6 |

### Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Calculus | 0.81 | 0.84 | 0.83 |
| Caries | 0.99 | 0.98 | 0.99 |
| Discoloration | 0.99 | 0.99 | 0.99 |
| Gingivitis | 0.86 | 0.93 | 0.89 |
| Hypodontia | 1.00 | 0.99 | 0.99 |
| Ulcer | 1.00 | 1.00 | 1.00 |

## Project Structure

```
oral-classify/
├── api/                    # FastAPI server + GradCAM
│   ├── server.py          # API endpoints
│   ├── inference.py       # Model loading & prediction
│   └── gradcam.py         # GradCAM++ visualization
├── src/
│   ├── models/            # Model architectures
│   ├── data/              # Dataset utilities
│   └── utils/             # Visualization tools
├── scripts/               # Training & evaluation scripts
├── configs/               # Configuration files
└── outputs/               # Trained weights (not in git)
```

## Quick Start

### Installation

```bash
conda activate cliniq
pip install -r requirements.txt
```

### Run Inference API

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8001
```

Then open: http://localhost:8001/docs

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classes` | GET | List disease classes |
| `/predict` | POST | Classify image (JSON response) |
| `/predict/gradcam` | POST | Classify with GradCAM heatmap overlay |

### Example Usage

```python
import requests

# Predict
with open("dental_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8001/predict",
        files={"file": f}
    )
    print(response.json())
# {'predicted_class': 'Caries', 'confidence': 0.99, ...}

# Get GradCAM visualization
with open("dental_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8001/predict/gradcam",
        files={"file": f}
    )
    with open("gradcam_result.jpg", "wb") as out:
        out.write(response.content)
```

## Training

```bash
cd scripts
python train_convnext.py
```

## GradCAM Visualization

GradCAM (Gradient-weighted Class Activation Mapping) shows which regions of the image the model focuses on for its prediction. This is crucial for:

- **Interpretability**: Understand model decisions
- **Validation**: Verify model is looking at correct areas
- **Trust**: Build confidence in AI predictions

## Classes

1. **Calculus** - Tartar buildup
2. **Caries** - Tooth decay/cavities
3. **Discoloration** - Tooth staining
4. **Gingivitis** - Gum inflammation
5. **Hypodontia** - Missing teeth
6. **Ulcer** - Oral sores

## Weights

Model weights are not included in the repository. Download or train:

```bash
# Train new model
python scripts/train_convnext.py

# Or download pre-trained (if available)
# Place in outputs/weights/best_model.pth
```
