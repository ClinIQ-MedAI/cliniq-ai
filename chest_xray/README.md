# 🫁 NIH Chest X-ray Multi-Label Classification

This project implements a robust deep learning pipeline for multi-label classification of 14 common thoracic diseases using the NIH Chest X-ray14 dataset.

## 🚀 Features

- **State-of-the-Art Model**: ConvNeXt-Large backbone pretrained on ImageNet.
- **Robust Training Pipeline**:
    - **Focal Loss**: To handle severe class imbalance.
    - **Data Augmentation**: CLAHE, Gaussian Noise, Geometric transforms.
    - **Mixed Precision (AMP)**: Faster training with less memory.
    - **Patient-Level Split**: Prevents data leakage between train/val sets.
- **Comprehensive Error Analysis**:
    - Per-class metrics (AUC, F1, Precision, Recall).
    - **Grad-CAM Visualization**: Explainability for False Positives/Negatives.
    - Metric Curves: ROC and Precision-Recall curves.

---

## 📊 Dataset & EDA

The dataset contains **112,120 Frontal X-rays** from **30,805 unique patients**.

### Class Imbalance
The dataset is heavily imbalanced:
- **No Finding**: ~54% of images.
- **Infiltration**: ~18% (Most common disease).
- **Hernia**: ~0.2% (Rarest).

> [!NOTE]
> We handle this using **Focal Loss** (gamma=2.0) and **Inverse Frequency Class Weights**.

### Image Preprocessing
- **Resize**: 512x512 pixels.
- **Normalization**: ImageNet stats.
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization to enhance bone/tissue contrast.

---

## 🏗️ Model Architecture

We use **ConvNeXt-Large** adapted for multi-label classification:
- **Input**: (B, 3, 512, 512)
- **Backbone**: ConvNeXt-Large features (1536 channels).
- **Head**: LayerNorm -> Dropout(0.2) -> Linear(1536, 14).
- **Activation**: **Sigmoid** (Independent probability per class).

---

## 🛠️ Usage

### 1. Training

Run the training script with default configuration:

```bash
python scripts/train.py
```

**Key Parameters** (in `config.py`):
- `EPOCHS`: 25
- `BATCH_SIZE`: 4 (or 32 with gradient accumulation)
- `LEARNING_RATE`: 1e-4 (Cosine Annealing)

### 2. Error Analysis & Evaluation

After training, run the analysis script to generate reports and visualizations:

```bash
python scripts/error_analysis.py --checkpoint outputs/checkpoints/best.pt
```

**Output**:
- `report.md`: Detailed metrics per class.
- `visualizations/`: Grad-CAM heatmaps for top False Positives and False Negatives.
- `curves/`: ROC and Precision-Recall plot images.

---

## 📈 Metric Curves

The error analysis script generates:
- **ROC Curves**: To evaluate trade-off between TPR and FPR.
- **Precision-Recall Curves**: Critical for imbalanced datasets (e.g., Hernia).

---

## 📁 Directory Structure

```
chest_xray/
├── config.py              # Configuration & Hyperparameters
├── data/
│   ├── dataset.py         # Dataset loader with patient-splits
│   └── transforms.py      # CLAHE & Augmentations
├── models/
│   └── convnext.py        # ConvNeXt model definition
├── scripts/
│   ├── train.py           # Main training loop
│   └── error_analysis.py  # Inference & Reporting
├── utils/
│   ├── losses.py          # Focal Loss implementation
│   └── visualization.py   # Grad-CAM & Plotting utils
└── eda/
    └── EDA_REPORT.md      # Detailed exploratory analysis
```

## 🔍 Explainability (Grad-CAM)

We use **Grad-CAM++** to visualize model attention.
- **Correct Predictions**: Heatmap should focus on the lung region relevant to the pathology.
- **Errors**:
    - **False Positive**: If heatmap focuses on medical devices (pacemakers, tubes), the model may have learned incorrect biases.
    - **False Negative**: If heatmap misses the lesion, it may be a resolution or contrast issue.

---

*Author: @Moabouag*
