# config.py
import os
import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data & training hyperparameters
DATA_DIR = "dataset"
BATCH = 32
NUM_CLASSES = 6
EPOCHS = 25
LR = 1e-4
LOG_EVERY = 10

# Normalization (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Paths for outputs
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_resnet50_oral.pth")

# Make sure folders exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
