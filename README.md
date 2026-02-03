# ClinIQ AI

Medical Imaging AI by ClinIQ-MedAI - Advanced deep learning solutions for clinical diagnostics.

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| [**chatbot-app/**](./chatbot-app/) | Healthcare Chatbot | ✅ Active |
| [**oral-xray/**](./oral-xray/) | Dental X-Ray Detection & Classification | ✅ Production |
| [**oral-classify/**](./oral-classify/) | Oral Disease Classification with GradCAM | ✅ Production |
| **dmri/** | Diffusion MRI Analysis | 🚧 Coming Soon |

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

### Classes
1. Calculus (tartar)
2. Caries (cavities)
3. Discoloration
4. Gingivitis
5. Hypodontia
6. Ulcer

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

## License

Proprietary - All rights reserved.
