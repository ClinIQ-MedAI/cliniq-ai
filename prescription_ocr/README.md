# Prescription OCR (TrOCR + PaddleOCR)

AI-powered **prescription text extraction** using **TrOCR** (Transformer OCR) fine-tuned on medical handwriting, combined with **PaddleOCR** and **Regex** for structured data extraction.

This feature provides:
- ✅ **End-to-end Pipeline** for drug name and dosage extraction
- ✅ **TrOCR Fine-tuning** scripts for handwritten medical text
- ✅ **Synthetic Data Generator** to create training samples
- ✅ **Regex-based NLP** for dosage/frequency parsing
- ✅ **Clean modular structure** (`pipeline/`, `training/`, `nlp/`)

---

## Model performance

> Reported run (TrOCR-Small fine-tuned on Synthetic Medical Data)

| Metric | Value |
|---|---:|
| **Character Error Rate (CER)** | **7.42%** |
| **Validation Loss** | **0.246** |
| **Architecture** | TrOCR (Vision Encoder-Decoder) |
| **Training Steps** | 8000 |

### Training Logs
- **Best Model Checkpoint**: Step 8000
- **Rapid Convergence**: Loss dropped from ~3.11 to ~0.18

---

## Pipeline Flow

1. **Preprocessing**: CLAHE enhancement + Median Blur Denoising
2. **Detection**: PaddleOCR detects text regions (Bounding Boxes)
3. **Recognition**: Fine-tuned TrOCR recognizes handwritten drug names
4. **NLP & Correction**: 
   - Fuzzy Matching against drug database
   - Regex extraction for Dosage, Frequency, and Duration
5. **Output**: Structured JSON

---

## Project structure

```text
prescription_ocr/
├── demo/                   # Interactive demo scripts
├── nlp/                    # NLP & Text Post-processing
│   ├── regex_extraction.py # Dosage/frequency pattern matching
│   └── llm_client.py       # Optional LLM integration
├── ocr/                    # OCR Engine Wrappers
├── pipeline/               # Main Pipeline Orchestration
├── training/               # Training & Fine-tuning Scripts
│   ├── synthetic_data_generator.py # Generate synthetic handwriting data
│   ├── trocr_finetune.py   # HuggingFace Trainer pipeline
│   └── plot_metrics.py     # Training visualization tools
├── utils/                  # Helper utilities (image loading, etc.)
├── config.py               # Global configuration parameters
└── main.py                 # Main CLI Entry Point
```

---

## Quick start

### 1) Installation

```bash
pip install -r requirements.txt
```

### 2) Usage (CLI)

#### Process a single image
```bash
python main.py --image path/to/prescription.jpg
```

#### Save output to JSON
```bash
python main.py --image prescription.png --output result.json
```

#### Run interactive demo
```bash
python main.py --demo
```

#### Batch Process a Directory
```bash
python main.py --batch ./data/raw_images/
```

---

## Weights

Model weights are managed via HuggingFace or local checkpoints.

### Option A — Train your own weights
1. **Generate Synthetic Data**:
   ```bash
   python training/synthetic_data_generator.py
   ```
   This will create a dataset in `training/synthetic_data/`.

2. **Run Training**:
   ```bash
   python training/trocr_finetune.py
   ```
   Artifacts will be saved to `training/logs/` and checkpoints to the configured output dir.

---

## Output Format

The tool generates a structured JSON response:

```json
{
  "status": "success",
  "medications": [
    {
      "drug": "Augmentin",
      "drug_corrected": "Augmentin",
      "dosage": "1g",
      "frequency": "1 tablet every 12 hours",
      "duration": "7 days",
      "confidence": 0.98
    }
  ],
  "metadata": {
      "processing_time": "1.2s",
      "ocr_model": "trocr-small-handwritten"
  }
}
```

---

## Visualization

When running with `--demo` or without `--no-visualization`, the pipeline saves an annotated image showing:
- **Red Boxes**: Detected text regions
- **Labels**: OCR recognized text overlaid on the image

---

## Troubleshooting

### CUDA Out of Memory (OOM)
- Reduce `batch_size` in `config.py` or `trocr_finetune.py`.
- Resize very large input images before processing.

### Poor Recognition Accuracy
- Ensure the input image matches the domain (medical prescriptions).
- Check preprocessing settings (contrast/brightness) in `config.py`.
- Add the specific drug name to `data/drugs_db.json` for better fuzzy matching.

---

## Disclaimer
This project is for research/educational use. It is **not** a medical device and must be clinically validated before any real medical usage.
