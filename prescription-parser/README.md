# Prescription Parser Service

AI-Powered Handwritten Prescription Parsing & Medication Normalization.

## Pipeline

```
Image → Qwen2-VL-72B (AWQ, vLLM) → JSON drugs/dosage/frequency → RapidFuzz vs Egyptian Drugs DB → verified JSON
```

## Components

- `prescription_pipeline.py` — `PrescriptionParserService` (data ingestion, vLLM init, fuzzy normalization).
- `api/server.py` — FastAPI server on port 8005.
- `requirements.txt` — extra deps on top of the shared `cliniq` env.

## Hardware

- NVIDIA RTX A6000 (48 GB VRAM)
- `Qwen/Qwen2-VL-72B-Instruct-AWQ` (4-bit) ≈ ~38–42 GB VRAM. Tight but fits.
- vLLM `gpu_memory_utilization=0.92`, `max_model_len=8192`.

## Standalone test

```bash
cd cliniq/prescription-parser
python prescription_pipeline.py /path/to/prescription.jpg
```

## API

```bash
# health
curl http://localhost:8005/health

# parse a prescription image
curl -F "file=@/path/to/rx.jpg" http://localhost:8005/parse | jq

# also exposed for the chatbot
curl -F "file=@/path/to/rx.jpg" http://localhost:8005/predict_for_llm | jq
```

## First run

The Egyptian medicines dataset and Qwen2-VL-72B-AWQ weights are downloaded
lazily on first request (~40 GB). Subsequent boots reuse the HF cache.
