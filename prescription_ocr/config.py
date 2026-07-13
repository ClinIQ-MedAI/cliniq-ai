"""
ClinIQ - OCR Prescription Feature Configuration
Centralized settings for all pipeline components.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
PROCESSED_IMAGES_DIR = DATA_DIR / "processed_images"
SAMPLES_DIR = DATA_DIR / "samples"
DRUGS_DB_PATH = DATA_DIR / "drugs_db.json"

# =============================================================================
# IMAGE PREPROCESSING SETTINGS
# =============================================================================
PREPROCESSING = {
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8),
    
    # Median Blur for denoising
    "blur_kernel_size": 3,
    
    # Whether to save intermediate processed images
    "save_processed": True,
}

# =============================================================================
# TROCR SETTINGS (Fine-tuned model)
# =============================================================================
TROCR = {
    "model_path": os.getenv("TROCR_MODEL_PATH", "/N/scratch/moabouag/trocr_drugs/checkpoint-8000"),
    "use_gpu": os.getenv("TROCR_USE_GPU", "true").lower() == "true",
}

# =============================================================================
# OCR SETTINGS
# =============================================================================
OCR = {
    # Backend selection: "easyocr" (recommended for Arabic) or "paddle"
    # Note: These backends are used for DETECTION. 
    # If use_trocr=True, TrOCR is used for RECOGNITION on the detected boxes.
    "backend": os.getenv("OCR_BACKEND", "paddle"), # Paddle often yields better detection boxes
    
    # Enable TrOCR refinement
    "use_trocr": True,
    
    # Languages for EasyOCR (list)
    "languages": os.getenv("OCR_LANGUAGES", "ar,en").split(","),
    
    # PaddleOCR language (single code)
    "lang": "ar",
    
    "use_angle_cls": True,  # Enable text angle classification
    "use_gpu": os.getenv("OCR_USE_GPU", "true").lower() == "true",  # Use GPU for faster processing
    
    # Detection settings (PaddleOCR)
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.5,
    
    # Recognition settings
    "rec_batch_num": 6,
    
    # Show logs
    "show_log": False,
}

# =============================================================================
# NLP / TEXT EXTRACTION SETTINGS
# =============================================================================
NLP = {
    # Fuzzy matching threshold (0-100, higher = stricter)
    "fuzzy_threshold": 80,
    
    # Dosage patterns to normalize
    "dosage_patterns": {
        r"1x1": "once daily",
        r"1x2": "twice daily", 
        r"1x3": "3 times per day",
        r"2x1": "once daily (2 tablets)",
        r"2x2": "twice daily (2 tablets)",
        r"2x3": "3 times per day (2 tablets)",
    },
}

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
VISUALIZATION = {
    "box_color": (0, 255, 0),  # Green in BGR
    "box_thickness": 2,
    "text_color": (0, 0, 255),  # Red in BGR
    "font_scale": 0.6,
    "save_visualization": True,
}

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
OUTPUT = {
    "json_indent": 2,
    "include_raw_text": True,
    "include_confidence": True,
}

# =============================================================================
# LLM SETTINGS (for prescription extraction)
# =============================================================================
LLM = {
    # API Configuration (same as chatbot-app)
    "api_base_url": os.getenv("API_BASE_URL", "https://llm.jetstream-cloud.org/api/"),
    "api_key": os.getenv("API_KEY", "sk-0b8d90d6b3a44c64a5816df734975a62"),
    "model": os.getenv("MODEL", "gpt-oss-120b"),
    
    # Generation settings
    "max_tokens": 800,
    "temperature": 0.2,  # Lower for more consistent structured output
}
