# ClinIQ AI

Medical Imaging AI by ClinIQ-MedAI - Advanced deep learning solutions for clinical diagnostics.

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| [**chatbot-app/**](./chatbot-app/) | Healthcare Chatbot with Arabic Support | ✅ Active |
| [**bone-detect/**](./bone-detect/) | Pediatric Wrist Fracture Detection | ✅ Training |
| [**oral-xray/**](./oral-xray/) | Dental X-Ray Detection & Classification | ✅ Production |
| [**oral-classify/**](./oral-classify/) | Oral Disease Classification with GradCAM | ✅ Production |
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

- **YOLOv11x** - State-of-the-art object detection
- **800px images** - High resolution for subtle fractures
- **Arabic + English APIs** - LLM-ready diagnosis output

### Classes Detected (TOP-3)
| Class | Description | Priority |
|-------|-------------|----------|
| **fracture** | Bone fractures | 🔴 HIGH |
| **metal** | Surgical hardware/implants | ℹ️ INFO |
| **periostealreaction** | Bone healing indicators | 🟡 MODERATE |

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
