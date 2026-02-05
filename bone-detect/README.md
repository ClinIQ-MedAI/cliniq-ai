# Bone Fracture Detection - GrazPedWri-DX

YOLOv11x-based pediatric wrist fracture detection trained on the GrazPedWri-DX dataset.

## 📊 Model Performance

### Per-Class AP@0.5

| Class | AP@0.5 | Description |
|-------|--------|-------------|
| **fracture** | **0.970** | Bone fractures - PRIMARY target |
| **metal** | 0.942 | Surgical hardware/implants |
| **periostealreaction** | 0.708 | Bone healing indicators |

### Overall Metrics

| Metric | Value |
|--------|-------|
| **mAP@0.5** | 0.875 |
| mAP@0.5:0.95 | 0.587 |
| Precision | 0.856 |
| Recall | 0.840 |
| Epochs | 100 |

---

## 🏗️ Model Architecture

| Parameter | Value |
|-----------|-------|
| **Model** | YOLOv11x (Ultralytics) |
| **Parameters** | ~56.9M |
| **Image Size** | 800×800 |
| **Batch Size** | 24 |
| **Optimizer** | AdamW (auto) |
| **Training Time** | ~26 hours (100 epochs) |

### Augmentation
- HSV: h=0.015, s=0.7, v=0.4
- Rotation: ±15°
- Translate: 0.1
- Scale: 0.5
- Flip: 50% horizontal + vertical
- Mosaic: 1.0
- MixUp: 0.1

---

## 📁 Project Structure

```
bone-detect/
├── api/
│   ├── __init__.py
│   └── server.py           # FastAPI with Arabic support
├── train_top3.py            # Main training script
├── resume_top3.py           # Resume training
├── data_top3.yaml           # Dataset config (TOP-3 classes)
├── data.yaml                # Full 8-class config
└── outputs/
    └── YOLO11x_TOP3_*/
        ├── weights/
        │   ├── best.pt      # Best model checkpoint
        │   └── last.pt      # Latest checkpoint
        └── results.csv      # Training metrics
```

---

## 🔌 API Endpoints

| Endpoint | Language | Description |
|----------|----------|-------------|
| `POST /predict` | EN | Full JSON diagnosis |
| `POST /predict_text` | EN | Plain text report |
| `POST /predict_for_llm` | EN | LLM-optimized JSON |
| `POST /predict_text_ar` | AR | تقرير التشخيص |
| `POST /predict_for_llm_ar` | AR | JSON للذكاء الاصطناعي |

### Example Response (`/predict_for_llm`)
```json
{
  "patient_context": "Pediatric patient with wrist pain/trauma",
  "modality": "X-ray",
  "body_part": "Wrist",
  "ai_findings": [
    {
      "finding": "fracture",
      "confidence": "92.3%",
      "severity": "HIGH",
      "clinical_meaning": "bone fracture requiring immediate medical attention"
    }
  ],
  "has_fracture": true,
  "urgency": "HIGH",
  "recommendations": [
    "URGENT: Fracture detected - recommend immediate orthopedic consultation",
    "Consider immobilization and pain management"
  ]
}
```

### Arabic Response (`/predict_for_llm_ar`)
```json
{
  "language": "ar",
  "patient_context": "مريض أطفال يعاني من ألم/إصابة في المعصم",
  "ai_findings": [
    {
      "finding": "كسر",
      "confidence": "92.3%",
      "severity": "عالي"
    }
  ],
  "recommendations": ["عاجل: تم اكتشاف كسر - يوصى باستشارة جراحة العظام فوراً"]
}
```

---

## 🚀 Quick Start

### Run API Server
```bash
cd bone-detect
python api/server.py
# API at http://localhost:8001
# Docs at http://localhost:8001/docs
```

### Run Inference
```bash
curl -X POST "http://localhost:8001/predict" \
  -F "file=@wrist_xray.jpg"
```

### Resume Training
```bash
python resume_top3.py
```

---

## 📦 Dataset

**GrazPedWri-DX** - Pediatric Wrist Radiograph Dataset

| Split | Images |
|-------|--------|
| Train | 17,100 |
| Validation | 3,227 |
| **Total** | 20,327 |

Original dataset has 8 classes, model trained on TOP-3 highest performing:
- fracture (best AP)
- metal
- periostealreaction

---

## 🛠️ Training

### Start Fresh Training
```bash
python train_top3.py
```

### Resume from Checkpoint
```bash
python resume_top3.py
```

### Monitor Progress
Training saves:
- `training_progress.png` - Live plots
- `metrics_history.json` - All metrics per epoch
- `results.csv` - YOLO training log

---

## 📚 References

- **Dataset**: [GrazPedWri-DX](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193)
- **Model**: [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- **Paper**: Nagy et al. "A pediatric wrist trauma X-ray dataset"

---

**ClinIQ-MedAI** - Medical AI Research & Development
