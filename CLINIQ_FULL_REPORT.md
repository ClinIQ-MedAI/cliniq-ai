# ClinIQ AI Platform — Full Technical Report

**Repository:** https://github.com/ClinIQ-MedAI/cliniq-ai  
**Team:** ClinIQ-MedAI  
**Type:** Graduation Project — Medical AI Platform  

---

## What is ClinIQ?

ClinIQ is a **complete medical AI stack** — not a single model demo.  
It's a production-grade platform with:

- 6 specialized AI models covering radiology + oral diagnostics
- 1 conversational assistant (bilingual AR/EN)
- A Flask gateway routing everything together
- An async job queue (Redis/RabbitMQ) connecting to external backends (.NET)
- GradCAM explainability for clinical trust
- LLM-ready JSON outputs from every service

---

## System Architecture

```
Patient (Browser)
       ↓
Flask Chatbot Gateway  (:5000)
       │
       ├── /api/chat   ──→  LLM (gpt-oss-120b) + FAQ + Appointment routing
       │
       └── /api/upload ──→  Modality Router
                             ├── bone      (:8001)  bone-detect
                             ├── dental_xray(:8002) oral-xray
                             ├── chest     (:8003)  chest_xray
                             ├── dental_photo(:8004) oral-classify
                             └── prescription(:8005) prescription-parser


ASYNC QUEUE (New):
.NET Backend ──[JobMessage]──→  Upstash Redis  ──→  AI Worker (inside each service)
.NET Backend ←─[ResultMessage]─ Upstash Redis  ←──  AI Worker
```

---

## Service Map

| Port | Service | Technology | Task |
|------|---------|-----------|------|
| 5000 | chatbot-app | Flask + SQLite | Gateway + Conversational AI |
| 8001 | bone-detect | FastAPI + YOLO | X-ray Detection |
| 8002 | oral-xray | FastAPI + YOLO + ConvNeXt | X-ray 2-Stage Detection |
| 8003 | chest_xray | FastAPI + ConvNeXt | Multi-label Classification |
| 8004 | oral-classify | FastAPI + ConvNeXt | Image Classification + GradCAM |
| 8005 | prescription-parser | FastAPI + Qwen2-VL | VLM OCR Parsing |

---

---

# AI Models — Detailed Report

---

## 1. Bone Detect (Port 8001)

**Task:** Object Detection  
**Model:** YOLOv11x (Ultralytics, ~56.9M parameters)  
**Input:** Pediatric wrist X-ray images  
**Training Data:** GrazPedWri-DX — 20,327 images  

### What it detects (3 classes)
| Class | Clinical Meaning | Severity |
|-------|----------------|---------|
| `fracture` | Bone fracture | HIGH — urgent consultation |
| `metal` | Surgical hardware/implants from prior surgery | INFO |
| `periostealreaction` | Bone healing indicator / pathology | MODERATE |

### Performance
| Metric | Value |
|--------|-------|
| **mAP@0.5** | **87.5%** |
| mAP@0.5:0.95 | 58.7% |
| Precision | 85.6% |
| Recall | 84.0% |
| Fracture AP@0.5 | **97.0%** |
| Metal AP@0.5 | 94.2% |
| Periosteal AP@0.5 | 70.8% |

### Key Technical Details
- Image size: 800×800
- Confidence threshold: 0.30 (tuned from 0.25)
- 16-bit X-ray normalization → 8-bit RGB pre-processing
- Returns `annotated_image_base64` (bounding boxes rendered with OpenCV)
- Arabic + English endpoints
- Queue: `cliniq:jobs:bone`

### Output (predict_for_llm)
```json
{
  "modality": "X-ray",
  "body_part": "Wrist",
  "detections": [{"class_name": "fracture", "confidence": 0.923, "severity": "HIGH"}],
  "annotated_image_base64": "...",
  "has_fracture": true,
  "urgency": "HIGH",
  "summary": "Detected: 1 fracture.",
  "recommendations": ["URGENT: Fracture detected - recommend immediate orthopedic consultation"]
}
```

---

## 2. Oral X-Ray (Port 8002)

**Task:** Two-Stage Pipeline — Detection → Classification Refinement  
**Models:**
- Stage 1: YOLOv8x (detection, ~1024px input)
- Stage 2: ConvNeXt-Large (crop classification, refines YOLO results)

**Input:** Panoramic dental X-ray images  
**Dataset:** 2,688 validation images / 18,639 instances  

### What it detects (9 classes)
| Class | Type | Severity |
|-------|------|---------|
| Decay | Disease | HIGH |
| Apical Periodontitis | Disease | HIGH |
| Missing Tooth | Finding | MEDIUM |
| Wisdom Tooth | Finding | LOW |
| Dental Filling | Restoration | LOW |
| Root Canal Filling | Restoration | LOW |
| Implant | Restoration | LOW |
| Porcelain Crown | Restoration | LOW |
| Ceramic Bridge | Restoration | LOW |

### Performance
| Stage | Metric | Value |
|-------|--------|-------|
| YOLO Detection | mAP50 | 91.4% |
| YOLO Detection | mAP50-95 | **72.0%** |
| YOLO Detection | Precision | 87.9% |
| YOLO Detection | Recall | 89.0% |
| ConvNeXt Classifier | Accuracy | **99.32%** (epoch 9) |

### Key Technical Details
- Two-stage: YOLO detects bounding boxes → ConvNeXt re-classifies each crop
- Classifier confidence threshold: 0.70 (only accepts refined label if >70% confident)
- 16-bit input normalization for DICOM-style X-rays
- Severity map per class (HIGH/MEDIUM/LOW)
- Returns `was_refined: true/false` per detection (shows if YOLO label was corrected)
- Queue: `cliniq:jobs:dental_xray`

### Output (predict_for_llm)
```json
{
  "modality": "dental_xray",
  "num_detections": 3,
  "detections": [
    {
      "class_name": "Decay",
      "confidence": 0.87,
      "severity": "HIGH",
      "was_refined": true,
      "yolo_class": "cavity",
      "bbox": [120, 80, 200, 160]
    }
  ],
  "annotated_image_base64": "...",
  "summary": "Decay: 1 occurrence(s), max conf 0.87"
}
```

---

## 3. Chest X-Ray (Port 8003)

**Task:** Multi-Label Classification (multiple diseases per image)  
**Model:** ConvNeXt-Large (ImageNet pretrained, ~197M parameters)  
**Input:** Frontal chest X-rays (PA/AP view), resized to 512×512  
**Dataset:** NIH Chest X-ray14 — 112,120 images, 30,805 unique patients  

### What it classifies (13 active classes)
| Class | AUC | Notable |
|-------|-----|---------|
| Consolidation | 0.968 | Lung infection |
| Emphysema | 0.961 | Chronic damage |
| Effusion | 0.934 | Fluid around lungs |
| Fibrosis | 1.000 | Lung scarring |
| Pneumothorax | ~0.99 | Collapsed lung |
| Atelectasis | 0.895 | Lung collapse |
| Cardiomegaly | 0.895 | Enlarged heart |
| Mass | — | Abnormal growth |
| Infiltration | — | Abnormal lung substance |
| Nodule | — | Small growth |
| Pleural Thickening | — | Scar tissue |
| Hernia | — | Rare (<0.2%) |
| No Finding | — | Normal (54% of dataset) |

### Key Technical Details
- **Focal Loss** (gamma=2.0) to handle class imbalance (54% "No Finding")
- **Patient-Level Split** to prevent data leakage between train/val
- **GradCAM visualization** shows model attention regions
- Mixed Precision (AMP) training
- CLAHE preprocessing for contrast enhancement
- Sigmoid activation per class (independent multi-label probabilities)
- Returns `gradcam_image_base64` when `include_gradcam=true`
- Queue: `cliniq:jobs:chest`

### Output (predict_for_llm)
```json
{
  "modality": "Chest X-ray (PA/AP view)",
  "body_part": "Thorax",
  "ai_findings": {
    "primary_diagnosis": "Effusion",
    "confidence": "78.3%",
    "severity": "HIGH",
    "clinical_meaning": "Fluid around the lungs (pleural effusion)"
  },
  "detected_conditions": [{"condition": "Effusion", "probability": "78.3%"}],
  "urgency": "HIGH",
  "gradcam_image_base64": "...",
  "summary": "AI detected effusion with 78.3% confidence."
}
```

---

## 4. Oral Classify (Port 8004)

**Task:** Single-Label Image Classification + GradCAM explainability  
**Model:** ConvNeXt-Small + GradCAM++ visualization  
**Input:** Intraoral photos (clinical photographs of mouth/teeth)  

### What it classifies (6 classes)
| Class | Meaning | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Calculus | Tartar buildup | 0.81 | 0.84 | 0.83 |
| Caries | Tooth decay / cavities | 0.99 | 0.98 | 0.99 |
| Discoloration | Tooth staining | 0.99 | 0.99 | 0.99 |
| Gingivitis | Gum inflammation | 0.86 | 0.93 | 0.89 |
| Hypodontia | Missing teeth | 0.99 | 0.99 | 0.99 |
| Ulcer | Oral sores | 0.99 | 0.99 | 0.99 |

### Performance
| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **94.83%** |
| Architecture | ConvNeXt-Small |

### Key Technical Details
- GradCAM++ heatmap overlay on output image
- Derives bounding box from GradCAM activation map (no explicit detection head)
- Severity determined per predicted class
- Arabic + English endpoints
- Returns GradCAM as `annotated_image_base64` (not a separate field)
- Queue: `cliniq:jobs:dental_photo`

### Output (predict_for_llm)
```json
{
  "modality": "intraoral_photo",
  "predicted_class": "Caries",
  "confidence": "95.2%",
  "severity": "HIGH",
  "annotated_image_base64": "...",
  "gradcam_applied": true,
  "recommendations": ["Schedule dental examination", "Avoid sugary foods"]
}
```

---

## 5. Prescription OCR — TrOCR Pipeline (prescription_ocr/)

**Task:** Handwritten prescription text extraction  
**Models:** TrOCR (Vision Encoder-Decoder) + PaddleOCR  
**Architecture:** Two-model pipeline  

### Pipeline Flow
```
Image
  ↓ CLAHE + Median Blur (preprocessing)
  ↓ PaddleOCR (detect text regions / bounding boxes)
  ↓ TrOCR fine-tuned (recognize handwritten drug names)
  ↓ Fuzzy matching against drug database
  ↓ Regex NLP (dosage / frequency / duration extraction)
  ↓ Structured JSON output
```

### Performance
| Metric | Value |
|--------|-------|
| **Character Error Rate (CER)** | **7.42%** |
| Validation Loss | 0.246 |
| Training Steps | 8,000 |
| Convergence | Loss from 3.11 → 0.18 |

### Output
```json
{
  "medications": [
    {
      "drug_name": "Paracetamol",
      "dosage": "500mg",
      "frequency": "3x daily",
      "duration": "5 days"
    }
  ]
}
```

---

## 6. Prescription Parser — VLM Pipeline (Port 8005)

**Task:** Vision-Language Model prescription parsing + Egyptian drug normalization  
**Model:** Qwen2-VL-72B-Instruct-AWQ (4-bit quantized, ~38-42 GB VRAM)  
**Hardware:** NVIDIA RTX A6000 (48 GB VRAM)  

### Pipeline Flow
```
Image
  ↓ Qwen2-VL-72B (AWQ via vLLM)
  ↓ Strict JSON extraction (drug / dosage / frequency / duration)
  ↓ RapidFuzz matching against Egyptian Drugs Database
  ↓ Verified output with official_match + confidence_score
```

### Features
- Lazy model loading (downloads ~40GB on first request, cached after)
- `/status` endpoint: reports live loading stages to frontend
- Egyptian drug database normalization (local brand names)
- Handles Arabic handwriting natively (multilingual VLM)
- Queue: `cliniq:jobs:prescription`

### Output (predict_for_llm)
```json
{
  "success": true,
  "image_type": "prescription",
  "ai_findings": {
    "primary_diagnosis": "تم استخراج 3 دواء (2 منها متحقق رسمياً)",
    "medications": [
      {
        "drug_name": "أوجمنتين",
        "dosage": "625mg",
        "frequency": "مرتين يومياً",
        "duration": "7 أيام",
        "official_match": "Augmentin 625mg",
        "confidence_score": 0.94
      }
    ]
  }
}
```

---

## 7. Chatbot Gateway (Port 5000)

**Task:** Patient-facing conversational AI + upload router  
**Stack:** Flask + SQLite + JS frontend  

### Capabilities
- **Appointment booking** — book with specific doctors, check queue position
- **FAQ answering** — clinic hours, insurance, policies
- **Medical image upload** — routes to correct AI service by modality
- **PDF analysis** — extract and summarize medical documents
- **LLM-powered responses** — gpt-oss-120b via Jetstream Cloud API

### Technical Features
- SQLite-backed chat history per patient (`chat_history.db`)
- Sidebar: create/switch/delete/clear conversations
- 45+ exception handlers with retry + exponential backoff
- Bilingual (Arabic RTL + English LTR) with i18n dictionary
- GradCAM image rendered in-chat
- Upload progress card with staged status updates
- Severity detection → auto-suggests appointment for HIGH severity
- Streaming responses (SSE via `stream_with_context`)

### Routes
| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Chat UI |
| `/api/chat` | POST | Conversational AI |
| `/api/upload` | POST | Image/PDF analysis |
| `/api/doctors` | GET | Available doctors |
| `/api/appointments/book` | POST | Book appointment |
| `/api/appointments/queue` | GET | Queue position |
| `/api/chat/history` | GET | Chat history |
| `/api/prescription/status` | GET | VLM loading status proxy |

---

---

# Redis Queue Integration (New Feature)

---

## Why This Was Added

The AI services were fully synchronous — the backend had to wait (sometimes 15+ seconds) for inference to complete.

**Solution:** Add an async job queue using Redis Streams (or RabbitMQ) as a communication bridge, so the backend can submit jobs without waiting and receive results when they're ready.

---

## Architecture

```
.NET Backend (Machine A)          Python cliniq (Machine B)
      │                                    │
      │   cliniq:jobs:bone                 │
      ├────────────────────────────────────►│
      │                                    │ Load YOLO model
      │                                    │ Run inference
      │   cliniq:results                   │
      │◄───────────────────────────────────┤
      │                                    │
```

**Upstash Redis** (cloud) is the shared message broker between the two machines.

---

## Message Schemas

### JobMessage (Backend → AI Worker)
```json
{
  "job_id": "abc123def456",
  "modality": "bone",
  "image_base64": "<base64 encoded image bytes>",
  "image_url": null,
  "patient_id": "patient_001",
  "options": {"include_gradcam": false},
  "reply_to": null,
  "enqueued_at": "2026-06-27T10:00:00Z"
}
```

### ResultMessage (AI Worker → Backend)
```json
{
  "job_id": "abc123def456",
  "modality": "bone",
  "status": "completed",
  "result": { "...full predict_for_llm output..." },
  "error": null,
  "patient_id": "patient_001",
  "worker": "hostname:bone",
  "duration_ms": 13402.69,
  "finished_at": "2026-06-27T10:00:13Z"
}
```

---

## Queue Channels

| Channel | Direction | Purpose |
|---------|-----------|---------|
| `cliniq:jobs:bone` | Backend → Worker | Bone X-ray jobs |
| `cliniq:jobs:dental_xray` | Backend → Worker | Dental panoramic jobs |
| `cliniq:jobs:chest` | Backend → Worker | Chest X-ray jobs |
| `cliniq:jobs:dental_photo` | Backend → Worker | Intraoral photo jobs |
| `cliniq:jobs:prescription` | Backend → Worker | Prescription parsing jobs |
| `cliniq:results` | Worker → Backend | All results (shared) |

---

## How to Enable

```bash
# Install messaging deps
pip install redis pika

# Set env vars (in .env)
QUEUE_BACKEND=redis
REDIS_CONNECTION=causal-leopard-72320.upstash.io:6379,password=YOUR_TOKEN,ssl=true

# Run any service — worker starts automatically
cd bone-detect && python api/server.py
# → [worker:bone] consuming 'cliniq:jobs:bone' -> results to 'cliniq:results'
```

**If `QUEUE_BACKEND` is not set → services behave exactly as before (HTTP only). Zero breaking change.**

---

## .NET Backend Integration (C#)

### Publish a job
```csharp
var redis = ConnectionMultiplexer.Connect(
    config.GetConnectionString("Redis")); // same string as above

await redis.GetDatabase().StreamAddAsync("cliniq:jobs:bone",
    new NameValueEntry[] {
        new("data", JsonSerializer.Serialize(new {
            job_id = Guid.NewGuid().ToString("N"),
            modality = "bone",
            image_base64 = Convert.ToBase64String(imageBytes),
            patient_id = patientId
        }))
    });
```

### Read results
```csharp
const string stream = "cliniq:results", group = "backend";
// Create group once (MKSTREAM creates stream if missing)
try { await db.StreamCreateConsumerGroupAsync(stream, group, "0-0", true); }
catch (RedisServerException e) when (e.Message.Contains("BUSYGROUP")) { }

var entries = await db.StreamReadGroupAsync(stream, group, "api-worker", count: 10);
foreach (var entry in entries)
{
    var result = JsonSerializer.Deserialize<ResultMessage>(entry["data"]);
    if (result.status == "completed") {
        // result.result contains full AI output
        // result.result["annotated_image_base64"] → show to patient
    }
    await db.StreamAcknowledgeAsync(stream, group, entry.Id);
}
```

---

## Testing the Queue

```bash
# 1. Check connection
python3 -m messaging.cli ping

# 2. Submit a job (simulates backend)
python3 -m messaging.cli enqueue --modality bone --image /path/to/xray.jpg

# 3. Listen for results (simulates backend consumer)
python3 -m messaging.cli listen --verbose
```

---

## Messaging Package Structure

```
messaging/
├── config.py           # Load config from env vars
├── connection.py       # Parse .NET-style Redis connection strings
├── schemas.py          # JobMessage + ResultMessage
├── base.py             # Abstract Broker interface
├── redis_broker.py     # Redis Streams (XADD/XREADGROUP/XACK)
├── rabbitmq_broker.py  # RabbitMQ (durable queues, manual ack)
├── factory.py          # get_broker() picks Redis or RabbitMQ
├── worker.py           # JobWorker (consume → infer → publish)
├── fastapi_integration.py  # attach_worker(app, modality, route)
├── cli.py              # Test CLI (ping/enqueue/listen)
└── requirements.txt    # redis + pika
```

---

---

# Model Files

## Locally Present (in repo / models/)

| Service | File | Size |
|---------|------|------|
| bone-detect | `models/bone-detect/YOLO11x_TOP3_20260203_0645/weights/best.pt` | 114 MB |
| oral-xray | `models/oral-xray/yolo_v8x_base_1024/weights/best.pt` | ~391 MB |
| oral-xray | `models/oral-xray/convnext_large_20260130_090637/weights/best.pt` | ~2.2 GB |
| oral-classify | `models/oral-classify/SOTA_FINAL_20251124_1300/best_model.pth` | ~189 MB |

## Requires Download (Google Drive)

Root folder: https://drive.google.com/open?id=1sqt6QCXp_3UmJrEmM8b9Fu0CRC4f9cb2

| Service | File | Size | Link |
|---------|------|------|------|
| chest_xray | `outputs/checkpoints/best.pt` | ~3.0 GB | https://drive.google.com/open?id=1XoU0ai5lL5zsES936eis3niOnQ6BFIK6 |

## Auto-Downloaded on First Use

| Service | Model | Size |
|---------|-------|------|
| prescription-parser | Qwen2-VL-72B-Instruct-AWQ (HuggingFace) | ~40 GB |

---

---

# Quick Start

## Prerequisites
- Linux with CUDA GPU (recommended)
- Python 3.10+
- Conda environment `cliniq`

## Run All Services

```bash
# Terminal 1 — Bone Detection
cd bone-detect && python api/server.py

# Terminal 2 — Dental X-Ray
cd oral-xray && python api/server.py

# Terminal 3 — Chest X-Ray
cd chest_xray && python api/server.py

# Terminal 4 — Oral Classify
cd oral-classify && python api/server.py

# Terminal 5 — Prescription Parser
cd prescription-parser && python api/server.py

# Terminal 6 — Chatbot (open http://127.0.0.1:5000)
cd chatbot-app && python app.py
```

## Enable Queue Integration (Optional)

```bash
# .env file in each service directory
QUEUE_BACKEND=redis
REDIS_CONNECTION=host:6379,password=TOKEN,ssl=true
```

## Test via HTTP

```bash
# Chat
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to book an appointment", "patient_id": "p1", "language_preference": "en"}'

# Upload bone X-ray
curl -X POST http://127.0.0.1:5000/api/upload \
  -F "file=@xray.jpg" \
  -F "patient_id=p1" \
  -F "image_type=bone"
```

## Test Queue

```bash
pip install -r messaging/requirements.txt
python3 -m messaging.cli ping
python3 -m messaging.cli enqueue --modality bone --image xray.jpg
python3 -m messaging.cli listen --verbose
```

---

## Clinical Safety Note

This system is **decision-support software**, not a licensed autonomous diagnostic tool.  
Every AI output must be reviewed by a qualified clinician.

---

*Built by ClinIQ-MedAI*
