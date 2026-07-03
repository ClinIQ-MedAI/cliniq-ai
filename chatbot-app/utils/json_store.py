import json
import threading
from config import Config

_cache = {}
_cache_lock = threading.Lock()

def load_json(filepath, use_cache=False):
    """Load JSON file, with optional caching for static files."""
    if use_cache:
        with _cache_lock:
            if filepath in _cache:
                return _cache[filepath]

    if filepath.exists():
        with open(filepath, 'r') as f:
            data = json.load(f)
            if use_cache:
                with _cache_lock:
                    _cache[filepath] = data
            return data
    return []

def save_json(filepath, data):
    """Save JSON file and optionally update cache."""
    with _cache_lock:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        if filepath in _cache:
            _cache[filepath] = data

def get_doctors():
    """Get doctors (cached)."""
    return load_json(Config.DOCTORS_FILE, use_cache=True)

def get_faqs():
    """Get FAQs (cached)."""
    return load_json(Config.FAQ_FILE, use_cache=True)

def get_appointments():
    """Get appointments (always fresh)."""
    return load_json(Config.APPOINTMENTS_FILE, use_cache=False)
