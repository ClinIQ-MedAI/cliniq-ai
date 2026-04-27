# ClinIQ AI Platform

Production-grade medical AI monorepo by ClinIQ-MedAI.

This repository combines a bilingual clinical assistant with a suite of deep-learning services for radiology and oral diagnostics, all orchestrated to work together in real-world patient workflows.

Repository: https://github.com/ClinIQ-MedAI/cliniq-ai

## Why This Repo Matters

ClinIQ AI is not a single model demo. It is a complete AI stack:

1. Conversational triage and guidance in Arabic and English.
2. Multi-modal upload handling (images, X-rays, and PDFs).
3. Specialized inference services for bone, oral, chest, and prescription analysis.
4. LLM-ready medical JSON outputs for downstream reporting.
5. Visual explainability and annotated outputs to improve clinical trust.

## Monorepo Map

| Module | Purpose | Status |
|---|---|---|
| [chatbot-app](chatbot-app) | Main bilingual patient-facing assistant (Flask + JS UI) | Active |
| [bone-detect](bone-detect) | Pediatric wrist fracture/object detection (YOLO) | Active |
| [oral-xray](oral-xray) | Two-stage panoramic dental pipeline (YOLO + ConvNeXt refinement) | Active |
| [oral-classify](oral-classify) | Intraoral image classification with GradCAM overlay | Active |
| [chest_xray](chest_xray) | Multi-label thoracic disease classification + GradCAM | Active |
| [prescription-parser](prescription-parser) | VLM-based handwritten prescription parsing + drug normalization | New |
| [prescription_ocr](prescription_ocr) | OCR/NLP prescription pipeline (legacy/parallel path) | Maintained |

## Major Feature Drop (Current Update)

### 1) Chatbot Became a Real Product Interface

- SQLite-backed persistent chat memory (`chatbot-app/data/chat_history.db`) with per-patient isolation.
- Sidebar conversation management: create, switch, delete, and clear current chat.
- Upload UX upgraded with guided scan-type chooser + manual upload mode.
- Auto-send on guided upload selection, while preserving manual paperclip flow.
- Rich upload progress card with staged status updates and prescription-status polling.
- In-chat rendering of annotated output images (including GradCAM when available).
- Appointment modal integrated with queue position check and booking feedback.
- Export chat and copy-message actions added for usability.

### 2) Full AR/EN UI Localization + Smart Language Behavior

- Frontend now uses centralized i18n dictionary for runtime UI translation.
- Language selector (`AR` / `EN`) updates interface text instantly.
- Quick actions send prompts in selected language (no hardcoded Arabic-only quick prompts).
- Backend resolves response language using message language first, then default fallback.
- Strict one-language policy prompts added to prevent mixed-language answers in one response.

### 3) Better Medical Service Interop

- Chatbot routes uploads by modality to service-specific APIs.
- Added support for dedicated prescription parser API.
- Normalized response envelopes for LLM-compatible reporting.

### 4) Inference Service Upgrades

#### Bone Detection (`bone-detect/api/server.py`)
- Centralized model path under root `models/` tree.
- Confidence threshold tuned from 0.25 to 0.30.
- Added 16-bit X-ray normalization to robust 8-bit RGB pre-processing.
- LLM endpoint now returns structured detections and `annotated_image_base64`.

#### Oral Classify (`oral-classify/api`)
- Default weights moved to consolidated root models layout.
- Added GradCAM-first LLM endpoint output with encoded overlay image.
- Explicitly marks detector-style boxes as empty for classifier outputs.
- Service port aligned to `8004`.

#### Oral X-ray (`oral-xray/api/server.py`)
- Reworked server as a two-stage dental pipeline wrapper.
- Uses YOLO detection + classifier refinement for richer diagnostic classes.
- Returns standardized detections + severity + annotated image base64.
- Normalizes 16-bit inputs and draws readable visual labels.

#### Chest X-ray (`chest_xray/api/inference.py`)
- Checkpoint loading updated with `weights_only=False` compatibility guard.
- GradCAM target callable fixed for both 1D and batched outputs.

#### Prescription Parser (`prescription-parser/`)
- New production module exposing VLM-driven prescription parsing.
- Qwen2-VL (transformers) with strict JSON extraction format.
- Egyptian drug database normalization using RapidFuzz.
- Live status endpoint to surface long first-load model stages.

## System Architecture

```text
Patient UI (Web)
		|
		v
Flask Chatbot Gateway (chatbot-app :5000)
		|-- /api/chat   --> LLM + appointment/FAQ routing
		|-- /api/upload --> Modality router
						 |-- bone-detect       (:8001)
						 |-- oral-xray         (:8002)
						 |-- chest_xray        (:8003)
						 |-- oral-classify     (:8004)
						 |-- prescription      (:8005)
```

## Service Ports

| Service | Port | Main Endpoint |
|---|---:|---|
| chatbot-app | 5000 | `/api/chat`, `/api/upload` |
| bone-detect | 8001 | `/predict_for_llm` |
| oral-xray | 8002 | `/predict_for_llm` |
| chest_xray | 8003 | `/predict_for_llm` |
| oral-classify | 8004 | `/predict_for_llm` |
| prescription-parser | 8005 | `/predict_for_llm`, `/parse`, `/status` |

## Model Artifacts (Google Drive)

The heavy checkpoints are intentionally stored outside GitHub and referenced here.
Root Drive folder: https://drive.google.com/open?id=1sqt6QCXp_3UmJrEmM8b9Fu0CRC4f9cb2

| Service | Artifact | Size | Drive Link |
|---|---|---:|---|
| Bone Detect | `YOLO11x_TOP3_20260203_0645/weights/best.pt` | ~110 MB | https://drive.google.com/open?id=1Oh_R0gZL8gilMKGjDlbiLNTWoMz4EhKx |
| Oral Classify | `SOTA_FINAL_20251124_1300/best_model.pth` | ~189 MB | https://drive.google.com/open?id=1JBsf0xjm6xI-A0ofxuu5ufFgf0sVbx4g |
| Oral X-ray Detector | `yolo_v8x_base_1024/weights/best.pt` | ~391 MB | https://drive.google.com/open?id=13VHP4sKBWCMJrF7dZskuUrUun9MnCe_C |
| Oral X-ray Refiner | `convnext_large_20260130_090637/weights/best.pt` | ~2.2 GB | https://drive.google.com/open?id=1oWcRQZvc6UR3iyJJrs8I7bYP_0xb5oVO |
| Chest X-ray | `outputs/checkpoints/best.pt` | ~3.0 GB | https://drive.google.com/open?id=1XoU0ai5lL5zsES936eis3niOnQ6BFIK6 |

## Screenshot Gallery Placeholders

Replace these with real product screenshots before external release.

1. Chat Home
![Chat Home](docs/assets/screenshots/chat-home.png)

2. Language Switching (AR/EN)
![Language Switching](docs/assets/screenshots/chat-language-toggle.png)

3. Scan Upload Type Chooser
![Upload Type Chooser](docs/assets/screenshots/chat-upload-type-chooser.png)

4. Live Upload Progress
![Upload Progress](docs/assets/screenshots/chat-upload-progress.png)

5. Appointment Modal + Queue
![Booking Modal](docs/assets/screenshots/chat-booking-modal.png)

6. Inference Visualization Output
![Annotated Result](docs/assets/screenshots/chat-annotated-result.png)

## Runbook (Local)

### Prerequisites

- Linux + CUDA-capable GPU recommended.
- Conda environment prepared (`cliniq`).
- Python dependencies installed per module.
- External LLM key configured for chatbot gateway.

### Minimal Bring-Up Order

```bash
# Terminal 1
cd bone-detect
python api/server.py

# Terminal 2
cd oral-xray
python api/server.py

# Terminal 3
cd chest_xray
python api/server.py

# Terminal 4
cd oral-classify
python api/server.py

# Terminal 5
cd prescription-parser
python api/server.py

# Terminal 6
cd chatbot-app
python app.py
```

Then open http://127.0.0.1:5000

## API Examples

### Chat

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
	-H "Content-Type: application/json" \
	-d '{"message":"I want to book an appointment","patient_id":"patient_demo","language_preference":"en"}'
```

### Upload (with optional user text)

```bash
curl -X POST http://127.0.0.1:5000/api/upload \
	-F "file=@/path/to/image.jpg" \
	-F "patient_id=patient_demo" \
	-F "image_type=dental_xray" \
	-F "language_preference=en" \
	-F "user_message=Please explain this scan"
```

### Prescription Parser Health/Status

```bash
curl http://127.0.0.1:8005/health
curl http://127.0.0.1:8005/status
```

## Clinical Safety Notes

- This system is decision-support software, not a licensed autonomous diagnostic authority.
- Predictions can be wrong; every report must be clinically reviewed.
- Use patient data handling policies and remove PHI from public artifacts.

## Suggested Release Checklist

1. Replace screenshot placeholders with actual UI captures.
2. Confirm all Drive links are public or team-accessible.
3. Pin environment files and exact package versions per service.
4. Add smoke tests for all service health endpoints.
5. Validate bilingual responses for all major intents.

## Team

Built by ClinIQ-MedAI.

For product collaboration and deployment support, open an issue in this repository.
