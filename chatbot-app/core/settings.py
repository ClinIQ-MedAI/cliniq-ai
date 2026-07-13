"""Shared config, constants, and optional-feature imports for the chatbot.

Every core module does `from core.settings import *`, so anything defined here
(config values, the Flask `app`, stdlib/3rd-party imports, feature flags) is
available everywhere without re-importing.
"""

import os
import io
import re
import json
import time
import base64
import sqlite3
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
from flask import Flask, request
from dotenv import load_dotenv

# --- optional: PDF text extraction ---
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️ pdfplumber not installed - PDF support disabled")

# --- optional: image handling ---
try:
    from PIL import Image, ImageDraw
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False
    print("⚠️ Pillow not installed - Image support disabled")

# --- shared imaging utils (DICOM decode + OOD gate) live at the repo root ---
try:
    import sys as _sys
    _CLINIQ_ROOT = str(Path(__file__).resolve().parents[2])
    if _CLINIQ_ROOT not in _sys.path:
        _sys.path.insert(0, _CLINIQ_ROOT)
    from imaging import normalize_medical_image, DicomError, check_input
    DICOM_SUPPORT = True
except Exception as _e:
    DICOM_SUPPORT = False
    print(f"⚠️ DICOM support disabled: {_e}")

# --- optional: RAG grounding (on by default when an index exists) ---
RAG_SUPPORT = False
_rag_grounding = None
_rag_sources = None
if os.getenv("RAG_ENABLED", "1") != "0":
    try:
        from rag import grounding_block as _rag_grounding, sources as _rag_sources
        RAG_SUPPORT = True
    except Exception as _e:
        print(f"⚠️ RAG grounding disabled: {_e}")

load_dotenv()

# --- LLM config ---
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://llm.jetstream-cloud.org/api/")
MODEL = os.getenv("MODEL", "gpt-oss-120b")

# --- medical AI service endpoints ---
BONE_DETECT_API = "http://127.0.0.1:8001"
ORAL_DETECT_API = "http://127.0.0.1:8002"     # dental X-ray (YOLO)
CHEST_XRAY_API = "http://127.0.0.1:8003"
ORAL_CLASSIFY_API = "http://127.0.0.1:8004"   # intraoral photo (ConvNeXt+GradCAM)
PRESCRIPTION_API = "http://127.0.0.1:8005"    # Qwen2-VL + Egyptian drugs DB

# --- Flask app (routes are registered in app.py) ---
app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB

# --- CORS ---
# The bundled UI is same-origin, but Flutter Web / any browser client served from
# another origin needs these. Native mobile ignores CORS entirely. Restrict with
# CORS_ORIGINS="https://app.example.com,https://admin.example.com" in production.
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")


@app.after_request
def _apply_cors(response):
    origin = request.headers.get("Origin")
    if CORS_ORIGINS == "*":
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif origin and origin in {o.strip() for o in CORS_ORIGINS.split(",")}:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/<path:_unused>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def _cors_preflight(_unused=""):
    # Answer the browser's preflight; _apply_cors above attaches the headers.
    return ("", 204)


# --- request / response logging ----------------------------------------------
# One clear line per /api/* call: method, path, status, and how long it took.
# For POST /api/* the JSON (or form fields) is logged too, so a "why is reply
# null" question can be traced to the exact request. Unhandled errors print the
# full traceback. All of this goes to stdout, which the sbatch tees into the
# SLURM log so it shows up in the terminal you tail.
import sys as _sys
import traceback as _traceback
from flask import g as _g


def _short(value, limit=300):
    s = value if isinstance(value, str) else repr(value)
    return s if len(s) <= limit else s[:limit] + f"...(+{len(s)-limit})"


@app.before_request
def _log_request_start():
    _g._t0 = time.time()
    if not request.path.startswith("/api/"):
        return
    body = ""
    if request.method in ("POST", "PUT", "PATCH"):
        if request.is_json:
            body = _short(request.get_json(silent=True))
        elif request.form:
            # multipart: show field names + file names, never the file bytes
            fields = {k: _short(v, 60) for k, v in request.form.items()}
            files = {k: f.filename for k, f in request.files.items()}
            body = _short({"fields": fields, "files": files})
    print(f"[req] --> {request.method} {request.full_path.rstrip('?')}"
          + (f"  body={body}" if body else ""), flush=True)


@app.after_request
def _log_request_end(response):
    if request.path.startswith("/api/"):
        dt = (time.time() - getattr(_g, "_t0", time.time())) * 1000
        print(f"[req] <-- {request.method} {request.path}  "
              f"{response.status_code}  {dt:.0f}ms", flush=True)
    return response


@app.errorhandler(Exception)
def _log_exception(exc):
    # Print the traceback, then re-raise so Flask's normal error handling
    # (debugger page / 500) still runs.
    print(f"[req] !!! {request.method} {request.path}  EXCEPTION: "
          f"{type(exc).__name__}: {exc}", flush=True)
    _traceback.print_exc(file=_sys.stdout)
    _sys.stdout.flush()
    raise exc

# --- uploads ---
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'dcm', 'dicom'}
ALLOWED_PDF_EXTENSIONS = {'pdf'}

# --- overlay caps (avoid cluttered visualizations) ---
OVERLAY_MAX_DETECTIONS_BY_IMAGE_TYPE = {'dental_photo': 30, 'dental_xray': 50, 'dental': 30, 'bone': 20}
OVERLAY_DEFAULT_MAX_DETECTIONS = 30

# --- probe the LLM endpoint once at startup ---
LLM_AVAILABLE = False
try:
    _r = requests.get(f"{API_BASE_URL}models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=5)
    if _r.status_code == 200:
        LLM_AVAILABLE = True
        print("✓ LLM API is accessible!")
    else:
        print("✗ LLM API returned error:", _r.status_code)
except Exception as _e:
    print(f"✗ Could not connect to LLM API: {_e}")

# --- data paths ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DOCTORS_FILE = DATA_DIR / "doctors.json"
APPOINTMENTS_FILE = DATA_DIR / "appointments.json"
FAQ_FILE = DATA_DIR / "faq.json"
CHAT_DB_FILE = DATA_DIR / "chat_history.db"
MAX_HISTORY_MESSAGES = 50
HISTORY_WINDOW_MESSAGES = 20
