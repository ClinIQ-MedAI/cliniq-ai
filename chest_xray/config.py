"""
Configuration loader for Chest X-ray Classification
Loads settings from .env file
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load .env file
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)


def get_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment."""
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 'yes')


def get_float(key: str, default: float = 0.0) -> float:
    """Get float from environment."""
    return float(os.getenv(key, default))


def get_int(key: str, default: int = 0) -> int:
    """Get int from environment."""
    return int(os.getenv(key, default))


@dataclass
class DataConfig:
    """Data configuration."""
    root_dir: str = os.getenv("DATA_ROOT", "/N/scratch/moabouag/cliniq/data/chest")
    train_list: str = os.getenv("TRAIN_LIST", "train_val_list.txt")
    test_list: str = os.getenv("TEST_LIST", "test_list.txt")
    labels_file: str = os.getenv("LABELS_FILE", "Data_Entry_2017.csv")
    
    # Classes (excluding Hernia)
    classes: List[str] = field(default_factory=lambda: [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Infiltration", 
        "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
    ])
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = os.getenv("MODEL_NAME", "convnext_large")
    pretrained: bool = get_bool("PRETRAINED", True)
    num_classes: int = 14  # Will be updated from DataConfig
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Training configuration."""
    image_size: int = get_int("IMAGE_SIZE", 512)
    batch_size: int = get_int("BATCH_SIZE", 4)
    learning_rate: float = get_float("LEARNING_RATE", 1e-4)
    weight_decay: float = get_float("WEIGHT_DECAY", 1e-5)
    epochs: int = get_int("EPOCHS", 25)
    label_smoothing: float = get_float("LABEL_SMOOTHING", 0.05)
    
    # Loss
    loss_type: str = os.getenv("LOSS_TYPE", "focal")
    focal_gamma: float = get_float("FOCAL_GAMMA", 2.0)
    
    # Scheduler
    scheduler: str = os.getenv("SCHEDULER", "cosine")
    warmup_epochs: int = get_int("WARMUP_EPOCHS", 2)
    
    # Gradient accumulation
    gradient_accumulation_steps: int = get_int("GRADIENT_ACCUMULATION_STEPS", 2)
    
    # Early stopping
    early_stopping_patience: int = get_int("EARLY_STOPPING_PATIENCE", 10)


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    device: str = os.getenv("DEVICE", "cuda")
    num_workers: int = get_int("NUM_WORKERS", 8)
    pin_memory: bool = get_bool("PIN_MEMORY", True)
    mixed_precision: bool = get_bool("MIXED_PRECISION", True)
    seed: int = get_int("SEED", 42)


@dataclass
class EMAConfig:
    """EMA configuration."""
    enabled: bool = get_bool("USE_EMA", True)
    decay: float = get_float("EMA_DECAY", 0.999)


@dataclass
class OutputConfig:
    """Output paths configuration."""
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "outputs/checkpoints")
    log_dir: str = os.getenv("LOG_DIR", "outputs/logs")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        """Update dependent configurations."""
        self.model.num_classes = self.data.num_classes
    
    def __repr__(self):
        return f"""
╔══════════════════════════════════════════════════════════════╗
║           NIH Chest X-ray Classification Config              ║
╠══════════════════════════════════════════════════════════════╣
║ Data:                                                        ║
║   Root: {self.data.root_dir:<50} ║
║   Classes: {self.data.num_classes:<47} ║
╠══════════════════════════════════════════════════════════════╣
║ Model:                                                       ║
║   Name: {self.model.name:<50} ║
║   Pretrained: {str(self.model.pretrained):<43} ║
╠══════════════════════════════════════════════════════════════╣
║ Training:                                                    ║
║   Image Size: {self.training.image_size:<44} ║
║   Batch Size: {self.training.batch_size:<44} ║
║   Learning Rate: {self.training.learning_rate:<40} ║
║   Epochs: {self.training.epochs:<48} ║
║   Loss: {self.training.loss_type:<50} ║
║   Mixed Precision: {str(self.hardware.mixed_precision):<38} ║
╚══════════════════════════════════════════════════════════════╝
"""


# Global config instance
config = Config()


if __name__ == "__main__":
    print(config)
    print(f"\nClasses ({config.data.num_classes}):")
    for i, cls in enumerate(config.data.classes, 1):
        print(f"  {i:2d}. {cls}")
