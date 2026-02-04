# ClinIQ - Medical AI Platform

AI-powered medical imaging and healthcare solutions.

---

## 🏥 Projects

| Project | Description | Model | Performance | API Port |
|---------|-------------|-------|-------------|----------|
| **bone-detect** | Pediatric wrist fracture detection | YOLOv11x | 800px, TOP-3 classes | 8001 |
| **oral-classify** | 6-class oral disease classification | ConvNeXt-Small | **94.8% accuracy** | 8001 |
| **oral-xray** | Oral disease detection (Caries, Ulcer, Gingivitis) | YOLOv11x | 400 epochs | 8002 |
| **chatbot-app** | Healthcare AI assistant with Arabic support | Gemini API | Multi-patient memory | 5000 |

---

## 📁 Structure

```
cliniq-ai/
├── bone-detect/            # Pediatric wrist fracture detection
│   ├── api/                # FastAPI with LLM + Arabic output
│   ├── train_top3.py       # YOLOv11x training
│   └── resume_top3.py      # Resume training
├── oral-classify/          # ConvNeXt oral disease classification
│   ├── api/                # FastAPI + GradCAM + LLM + Arabic
│   └── scripts/            # Training scripts
├── oral-xray/              # YOLO oral detection
│   └── api/                # FastAPI with LLM + Arabic output
└── chatbot-app/            # Healthcare AI chatbot (Arabic support)
    └── app.py              # Flask + Gemini
```

---

## 🔌 API Endpoints

All APIs support **English and Arabic** output for LLM report generation:

| Endpoint | Language | Returns |
|----------|----------|---------|
| `POST /predict` | EN | Full JSON diagnosis |
| `POST /predict_text` | EN | Plain text report |
| `POST /predict_for_llm` | EN | LLM-optimized JSON |
| `POST /predict_text_ar` | **AR** | تقرير التشخيص |
| `POST /predict_for_llm_ar` | **AR** | JSON للذكاء الاصطناعي |

### Example Arabic Response (`/predict_for_llm_ar`)
```json
{
  "language": "ar",
  "patient_context": "مريض أطفال يعاني من ألم في المعصم",
  "ai_findings": {
    "finding": "كسر",
    "confidence": "87.5%",
    "severity": "عالي"
  },
  "recommendations": ["عاجل: تم اكتشاف كسر - يوصى باستشارة جراحة العظام فوراً"]
}
```

---

## 🚀 Quick Start

```bash
# Bone Detection API (port 8001)
cd bone-detect && python api/server.py

# Oral Classification API (port 8001)
cd oral-classify && python -m api.server

# Oral Detection API (port 8002)
cd oral-xray && python api/server.py

# Healthcare Chatbot (port 5000)
cd chatbot-app && python app.py
```

---

## 📊 Model Performance

### Bone Detection (bone-detect)
| Class | Status | Description |
|-------|--------|-------------|
| **fracture** | Primary | Bone fractures - HIGH priority |
| **metal** | Info | Surgical hardware/implants |
| **periostealreaction** | Moderate | Bone healing indicators |

### Oral Classification (oral-classify)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Calculus | 0.81 | 0.84 | 0.83 |
| **Caries** | 0.99 | 0.98 | 0.99 |
| **Discoloration** | 0.99 | 0.99 | 0.99 |
| Gingivitis | 0.86 | 0.93 | 0.89 |
| **Hypodontia** | 1.00 | 0.99 | 0.99 |
| **Ulcer** | 1.00 | 1.00 | 1.00 |

---

## 🤖 Healthcare Chatbot

Features:
- ✅ Arabic & English support
- ✅ Multi-patient conversation memory
- ✅ Patient isolation (secure contexts)
- ✅ Appointment scheduling
- ✅ Doctor directory
- ✅ FAQ system

---

**ClinIQ-MedAI - One platform for medical AI! 🏥🤖**
