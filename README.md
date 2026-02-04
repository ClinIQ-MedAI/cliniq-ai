# ClinIQ - Medical AI Platform

AI-powered medical imaging and healthcare solutions.

## 🏥 Projects

| Project | Description | Model | Performance |
|---------|-------------|-------|-------------|
| **oral-classify** | 6-class oral disease classification | ConvNeXt-Small | **94.8% accuracy** |
| **oral-xray** | Oral disease detection (Caries, Ulcer, Gingivitis) | YOLOv11x | 400 epochs |
| **chatbot-app** | Healthcare AI assistant with Arabic support | Gemini API | Multi-patient memory |

## 📁 Structure

```
cliniq-ai/
├── oral-classify/          # ConvNeXt oral disease classification
│   ├── api/                # FastAPI + GradCAM + LLM output
│   └── scripts/            # Training scripts
├── oral-xray/              # YOLO oral detection
│   └── api/                # FastAPI with LLM output
└── chatbot-app/            # Healthcare AI chatbot
    └── app.py              # Flask + Gemini
```

## 🔌 API Endpoints

All APIs have LLM-ready endpoints for report generation:

| Endpoint | Returns | Use Case |
|----------|---------|----------|
| `POST /predict` | Full JSON | Structured data |
| `POST /predict_text` | Plain text | Direct LLM input |
| `POST /predict_for_llm` | LLM-optimized JSON | Report generation |

### Example Response
```json
{
  "patient_context": "Dental examination",
  "ai_findings": {
    "primary_diagnosis": "Caries",
    "confidence": "97.5%",
    "severity": "HIGH"
  },
  "recommendations": ["Schedule restorative appointment"]
}
```

## 🚀 Quick Start

```bash
# Oral Classification API
cd oral-classify && python -m api.server

# Oral Detection API  
cd oral-xray && python api/server.py

# Healthcare Chatbot
cd chatbot-app && python app.py
```

## 📊 Model Performance

### Oral Classification
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Calculus | 0.81 | 0.84 | 0.83 |
| Caries | 0.99 | 0.98 | 0.99 |
| Discoloration | 0.99 | 0.99 | 0.99 |
| Gingivitis | 0.86 | 0.93 | 0.89 |
| Hypodontia | 1.00 | 0.99 | 0.99 |
| Ulcer | 1.00 | 1.00 | 1.00 |

---

**ClinIQ-MedAI - One platform for medical AI! 🏥🤖**
