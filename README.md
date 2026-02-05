# ClinIQ AI

Medical Imaging AI by ClinIQ-MedAI - Advanced deep learning solutions for clinical diagnostics.

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| [**chatbot-app/**](./chatbot-app/) | Healthcare Chatbot with Arabic Support | ✅ Active |
| [**bone-detect/**](./bone-detect/) | Pediatric Wrist Fracture Detection | ✅ Production |
| [**oral-xray/**](./oral-xray/) | Dental X-Ray Detection & Classification | ✅ Production |
| [**oral-classify/**](./oral-classify/) | Oral Disease Classification with GradCAM | ✅ Production |
| [**chest_xray/**](./chest_xray/) | Chest X-ray Multi-Label Classification | ✅ Production |
| [**prescription_ocr/**](./prescription_ocr/) | Prescription Handwriting OCR | ✅ Production |
| **dmri/** | Diffusion MRI Analysis | 🚧 Coming Soon |

---

## Healthcare Chatbot

**Location:** [`chatbot-app/`](./chatbot-app/)

AI-powered healthcare assistant with multi-language support:

- **Gemini API** - Advanced conversational AI
- **Arabic & English** - Full bilingual support
- **Multi-Patient Memory** - Isolated conversation contexts
- **Appointment Scheduling** - Doctor directory integration

### Features
- 🌍 Arabic language support (RTL interface)
- 👥 Patient isolation (secure contexts)
- 📅 Appointment scheduling
- 👨‍⚕️ Doctor directory
- ❓ FAQ system

### Quick Start
```bash
cd chatbot-app
python app.py
# Open http://localhost:5000
```

---

## Bone Fracture Detection

**Location:** [`bone-detect/`](./bone-detect/)

Pediatric wrist trauma detection using GrazPedWri-DX dataset:

- **YOLOv11x** - 56.9M parameters, state-of-the-art detection
- **mAP@0.5: 0.875** - High precision fracture detection
- **Arabic + English APIs** - LLM-ready diagnosis output

### Model Performance
| Class | AP@0.5 | Description |
|-------|--------|-------------|
| **fracture** | **0.970** | Bone fractures - PRIMARY 🔴 |
| **metal** | 0.942 | Surgical hardware/implants |
| **periostealreaction** | 0.708 | Bone healing indicators |

*Overall mAP@0.5: 0.875 | Precision: 0.856 | Recall: 0.840*

### API Endpoints
```
POST /predict          → JSON diagnosis
POST /predict_text     → Plain text report
POST /predict_for_llm  → LLM-optimized output
POST /predict_text_ar  → تقرير التشخيص بالعربية
POST /predict_for_llm_ar → JSON للذكاء الاصطناعي
```

### Quick Start
```bash
cd bone-detect
python api/server.py
# API at http://localhost:8001
```

---

## Oral X-Ray Detection & Classification

**Location:** [`oral-xray/`](./oral-xray/)

End-to-end pipeline for dental panoramic X-ray analysis:

- **YOLO v8x Detection** - 9 dental condition classes, mAP50-95: 0.707
- **ConvNeXt-Large Classification** - 99.32% accuracy on crop refinement
- **Production API** - FastAPI server for real-time inference

### Classes Detected
1. Apical Periodontitis
2. Decay
3. Wisdom Tooth
4. Missing Tooth
5. Dental Filling
6. Root Canal Filling
7. Implant
8. Porcelain Crown
9. Ceramic Bridge

### Quick Start
```bash
cd oral-xray
conda run -n cliniq python scripts/run_inference.py --image path/to/xray.jpg --visualize
```

---

## Oral Disease Classification

**Location:** [`oral-classify/`](./oral-classify/)

AI-powered oral disease classification with GradCAM visualization:

- **ConvNeXt-Small** - 94.83% accuracy, 189MB model
- **GradCAM++** - Visual explanations of predictions
- **FastAPI Server** - REST API for production deployment
- **Arabic Support** - Full bilingual diagnosis output

### Classes
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Calculus | 0.81 | 0.84 | 0.83 |
| Caries | 0.99 | 0.98 | 0.99 |
| Discoloration | 0.99 | 0.99 | 0.99 |
| Gingivitis | 0.86 | 0.93 | 0.89 |
| Hypodontia | 1.00 | 0.99 | 0.99 |
| Ulcer | 1.00 | 1.00 | 1.00 |

### Quick Start
```bash
cd oral-classify
uvicorn api.server:app --host 0.0.0.0 --port 8001
# Open http://localhost:8001/docs
```

---

## Chest X-ray Multi-Label Classification

**Location:** [`chest_xray/`](./chest_xray/)

Robust multi-label classification for 14 common thoracic diseases:

- **ConvNeXt-Large** - State-of-the-art backbone pretrained on ImageNet
- **Focal Loss** - Handles severe class imbalance (e.g., Hernia vs No Finding)
- **Grad-CAM++** - Visual explainability for False Positives/Negatives
- **Patient-Level Split** - Prevents data leakage
- **[Download Best Model Weights](https://drive.google.com/file/d/1pNuJt5Jm_V5X4-PoGTZsV1Bq70oWnAmM/view?usp=sharing)** - Pretrained ConvNeXt Checkpoint (3GB)

### Diseases Detected
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax.

### Quick Start
```bash
cd chest_xray
python scripts/error_analysis.py --checkpoint outputs/checkpoints/best.pt
```

### Model Performance (Validation)
| Class | AUC | F1-Score | Precision | Recall |
|-------|-----|----------|-----------|--------|
| **Atelectasis** | 0.89 | 0.00 | 0.00 | 0.00 |
| **Consolidation** | **0.97** | 0.15 | 0.08 | 1.00 |
| **Effusion** | **0.93** | **0.67** | 0.70 | 0.64 |
| **Emphysema** | **0.96** | **0.80** | 1.00 | 0.67 |
| **Infiltration** | 0.60 | 0.17 | 0.11 | 0.40 |
| **Mass** | 0.82 | 0.33 | 0.40 | 0.29 |
| **Nodule** | 0.88 | 0.44 | 0.29 | **0.90** |
| **Pneumothorax** | **0.99** | **0.67** | 1.00 | 0.50 |

*Note: Metrics calculated on the validation set.*

---

## Prescription OCR

**Location:** [`prescription_ocr/`](./prescription_ocr/)

End-to-end pipeline for structured medication extraction from handwritten prescriptions:

- **State-of-the-art Accuracy** - **7.42% CER** (Character Error Rate) on medical handwriting
- **TrOCR** - Fine-tuned Transformer-based recognition
- **PaddleOCR** - Robust text detection in complex layouts
- **NLP Engine** - Structured extraction of Dosage, Frequency, and Duration

### Pipeline
1. **Preprocessing**: CLAHE & Bilateral filtering
2. **Detection**: PaddleOCR locates text lines
3. **Recognition**: TrOCR converts images to text
4. **Extraction**: Regex & Fuzzy matching parses structured JSON

### Performance Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **Character Error Rate (CER)** | **7.42%** | Standard metric for handwriting OCR accuracy |
| **Validation Loss** | 0.162 | Low loss indicates robust generalization |
| **Training Loss** | 0.163 | No overfitting observed (close to Val Loss) |

### Quick Start
```bash
cd prescription_ocr
python main.py --image data/sample.jpg
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/ClinIQ-MedAI/cliniq-ai.git
cd cliniq-ai

# Create conda environment
conda env create -f oral-xray/environment.yml
conda activate cliniq
```

## Team

**ClinIQ-MedAI** - Medical AI Research & Development
