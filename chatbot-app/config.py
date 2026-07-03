import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    
    # LLM Settings
    API_KEY = os.getenv("API_KEY")
    API_BASE_URL = os.getenv("API_BASE_URL", "https://llm.jetstream-cloud.org/api/")
    MODEL = os.getenv("MODEL", "gpt-oss-120b")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Medical API endpoints
    BONE_DETECT_API = os.getenv("BONE_DETECT_API", "http://127.0.0.1:8001")
    ORAL_DETECT_API = os.getenv("ORAL_DETECT_API", "http://127.0.0.1:8002")
    CHEST_XRAY_API = os.getenv("CHEST_XRAY_API", "http://127.0.0.1:8003")
    ORAL_CLASSIFY_API = os.getenv("ORAL_CLASSIFY_API", "http://127.0.0.1:8004")
    PRESCRIPTION_API = os.getenv("PRESCRIPTION_API", "http://127.0.0.1:8005")
    
    # File handling
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64MB max file size
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    ALLOWED_PDF_EXTENSIONS = {'pdf'}
    
    # Chat History
    MAX_HISTORY_MESSAGES = 50
    HISTORY_WINDOW_MESSAGES = 20
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DOCTORS_FILE = DATA_DIR / "doctors.json"
    APPOINTMENTS_FILE = DATA_DIR / "appointments.json"
    FAQ_FILE = DATA_DIR / "faq.json"
    CHAT_DB_FILE = DATA_DIR / "chat_history.db"
    
    # Overlay controls
    OVERLAY_MAX_DETECTIONS_BY_IMAGE_TYPE = {
        'dental_photo': 30,
        'dental_xray': 50,
        'dental': 30,
        'bone': 20,
    }
    OVERLAY_DEFAULT_MAX_DETECTIONS = 30
    
    # Validation & Limits
    MAX_SLOTS_PER_TIME = 4
    MAX_PATIENT_NAME_LENGTH = 100
