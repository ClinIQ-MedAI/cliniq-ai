#!/usr/bin/env python3
"""
Healthcare Chatbot Flask Application
Handles appointment booking, doctor availability, general health questions,
and medical image/PDF analysis.
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import json
import re
import os
import io
import base64
import sqlite3
from datetime import datetime
from pathlib import Path
import requests
import numpy as np
from dotenv import load_dotenv
import time

# Optional imports for file processing
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️ pdfplumber not installed - PDF support disabled")

try:
    from PIL import Image, ImageDraw
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False
    print("⚠️ Pillow not installed - Image support disabled")

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://llm.jetstream-cloud.org/api/")
MODEL = os.getenv("MODEL", "gpt-oss-120b")

# Medical API endpoints
BONE_DETECT_API = "http://127.0.0.1:8001"
ORAL_DETECT_API = "http://127.0.0.1:8002"   # YOLO detection for dental X-rays
CHEST_XRAY_API = "http://127.0.0.1:8003"
ORAL_CLASSIFY_API = "http://127.0.0.1:8004"  # ConvNeXt+GradCAM for intraoral photos
PRESCRIPTION_API = "http://127.0.0.1:8005"   # Qwen2-VL-72B AWQ + Egyptian drugs DB

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_PDF_EXTENSIONS = {'pdf'}

# Overlay controls to avoid cluttered visualization.
OVERLAY_MAX_DETECTIONS_BY_IMAGE_TYPE = {
    'dental_photo': 30,
    'dental_xray': 50,
    'dental': 30,
    'bone': 20,
}
OVERLAY_DEFAULT_MAX_DETECTIONS = 30

# Try to test API connection
LLM_AVAILABLE = False
try:
    response = requests.get(f"{API_BASE_URL}models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=5)
    if response.status_code == 200:
        LLM_AVAILABLE = True
        print("✓ LLM API is accessible!")
    else:
        print("✗ LLM API returned error:", response.status_code)
except Exception as e:
    print(f"✗ Could not connect to LLM API: {e}")

# Data paths
DATA_DIR = Path(__file__).parent / "data"
DOCTORS_FILE = DATA_DIR / "doctors.json"
APPOINTMENTS_FILE = DATA_DIR / "appointments.json"
FAQ_FILE = DATA_DIR / "faq.json"
CHAT_DB_FILE = DATA_DIR / "chat_history.db"
MAX_HISTORY_MESSAGES = 50
HISTORY_WINDOW_MESSAGES = 20


def _get_chat_db_connection():
    """Create a SQLite connection for chat history operations."""
    conn = sqlite3.connect(CHAT_DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_chat_db():
    """Initialize chat history database if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _get_chat_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_patient_id_id
            ON chat_messages (patient_id, id)
            """
        )
        conn.commit()


def save_chat_message(patient_id: str, role: str, content: str):
    """Persist one chat message for a patient."""
    if not patient_id:
        patient_id = "anonymous"
    if not content:
        return

    with _get_chat_db_connection() as conn:
        conn.execute(
            "INSERT INTO chat_messages (patient_id, role, content) VALUES (?, ?, ?)",
            (patient_id, role, content),
        )
        conn.commit()


def get_chat_messages(patient_id: str, limit: int = None):
    """Load chat history for a patient in chronological order."""
    if not patient_id:
        patient_id = "anonymous"

    with _get_chat_db_connection() as conn:
        if isinstance(limit, int) and limit > 0:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE patient_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (patient_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE patient_id = ?
                ORDER BY id ASC
                """,
                (patient_id,),
            ).fetchall()

    return [{"role": row["role"], "content": row["content"]} for row in rows]


def clear_chat_messages(patient_id: str):
    """Delete all messages for one patient chat."""
    if not patient_id:
        patient_id = "anonymous"

    with _get_chat_db_connection() as conn:
        conn.execute("DELETE FROM chat_messages WHERE patient_id = ?", (patient_id,))
        conn.commit()


def trim_chat_messages(patient_id: str, max_messages: int = MAX_HISTORY_MESSAGES):
    """Keep only the latest N messages for a patient."""
    if not patient_id:
        patient_id = "anonymous"
    if not isinstance(max_messages, int) or max_messages <= 0:
        return

    with _get_chat_db_connection() as conn:
        conn.execute(
            """
            DELETE FROM chat_messages
            WHERE patient_id = ?
              AND id NOT IN (
                SELECT id
                FROM chat_messages
                WHERE patient_id = ?
                ORDER BY id DESC
                LIMIT ?
              )
            """,
            (patient_id, patient_id, max_messages),
        )
        conn.commit()


init_chat_db()


def load_json(filepath):
    """Load JSON file."""
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return []


def save_json(filepath, data):
    """Save JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def clean_markdown(text: str):
    """Remove markdown formatting from text for JSON compatibility."""
    import re
    
    # Remove bold (**text**)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # Remove italic (*text*)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove markdown headers (# text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Handle numbered lists - match space before number, then number, period, space
    # This handles cases like "word 1. symptom 2. another" -> "word\n1. symptom\n2. another"
    text = re.sub(r'\s+(\d+)\.\s+', r'\n\1. ', text)
    
    # Convert markdown lists to plain text with bullet points
    text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
    # Remove markdown table formatting (|, dashes, etc.)
    text = re.sub(r'\|[-\s|]+\|', '', text)  # Remove separator rows
    text = re.sub(r'\|\s*', '', text)  # Remove pipe characters
    # Clean up multiple line breaks (but preserve intentional ones)
    text = re.sub(r'\n\s*\n+', '\n', text)
    # Remove leading/trailing whitespace from lines
    text = '\n'.join(line.strip() for line in text.split('\n'))
    # Clean up any HTML entities
    text = text.replace('‑', '-')  # Replace special dash with regular dash
    text = text.replace('°', ' degrees')  # Replace degree symbol
    
    return text.strip()


def normalize_language_code(language):
    """Normalize UI/backend language codes to 'ar' or 'en'."""
    code = str(language or 'ar').strip().lower()
    if code.startswith('en'):
        return 'en'
    return 'ar'


def detect_text_language(text: str):
    """Lightweight language detection for Arabic/English user text."""
    content = str(text or '').strip()
    if not content:
        return None

    arabic_matches = re.findall(r'[\u0600-\u06FF]', content)
    english_matches = re.findall(r'[A-Za-z]', content)

    if arabic_matches and not english_matches:
        return 'ar'
    if english_matches and not arabic_matches:
        return 'en'
    if arabic_matches and english_matches:
        return 'ar' if len(arabic_matches) >= len(english_matches) else 'en'

    return None


def resolve_response_language(user_text: str = None, preferred_language: str = 'ar'):
    """Use message language when available; otherwise fallback to user preference."""
    detected = detect_text_language(user_text)
    if detected:
        return detected
    return normalize_language_code(preferred_language)


def get_language_policy(target_language: str = 'ar'):
    """Return strict language policy text for prompts."""
    code = normalize_language_code(target_language)
    if code == 'en':
        return (
            "Language policy (strict): Respond in English only. "
            "Do not include Arabic words, Arabic sentences, or bilingual duplicates."
        )
    return (
        "سياسة اللغة (إلزامي): أجب بالعربية فقط. "
        "لا تضف ترجمة إنجليزية ولا تكتب ردًا ثنائي اللغة."
    )


def build_upload_guidance_message(language: str = 'ar'):
    """Return upload guidance in Arabic or English."""
    if normalize_language_code(language) == 'en':
        return (
            "Great! 📷 You can upload files in two ways:\n"
            "1) Choose the type from the quick upload buttons shown below this message.\n"
            "2) Or upload manually using the 📎 button next to the input box.\n\n"
            "Supported file types:\n"
            "🩻 Dental X-ray\n"
            "🦴 Bone X-ray\n"
            "🫁 Chest X-ray\n"
            "🦷 Dental Photo\n"
            "📝 Prescription (image or PDF)\n"
            "📄 Medical PDF Report"
        )

    return (
        "ممتاز! 📷 تقدر ترفع الملف بطريقتين:\n"
        "1) اختار النوع من أزرار الرفع السريع اللي هتظهر تحت الرسالة.\n"
        "2) أو ارفع يدويًا من زر 📎 بجانب مربع الكتابة.\n\n"
        "أنواع الملفات المدعومة:\n"
        "🩻 أشعة أسنان\n"
        "🦴 أشعة عظام\n"
        "🫁 أشعة صدر\n"
        "🦷 صور أسنان (Photo)\n"
        "📝 روشتة طبية (صورة أو PDF)\n"
        "📄 تقارير طبية PDF"
    )


def make_api_request_with_retry(payload, headers, max_retries=3, initial_wait=1):
    """Make API request with automatic retry on failure."""
    is_stream = payload.get('stream', False)
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_BASE_URL}chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
                stream=is_stream
            )
            
            if response.status_code == 200:
                return response
            
            # If not 200, try to get error message
            error_msg = f"API Error {response.status_code}"
            try:
                error_msg += f": {response.text[:200]}"
            except:
                pass
                
            print(f"⚠️ {error_msg} (attempt {attempt + 1}/{max_retries})")
            
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                time.sleep(wait_time)
                continue
            
            return None
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                print(f"⚠️ Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return None
    
    return None


def get_llm_response(prompt: str, system_message: str = None, history: list = None):
    """Get response from LLM API using direct HTTP requests (Streaming)."""
    if not LLM_AVAILABLE:
        yield "I'm unable to access the LLM service at the moment. Please try booking an appointment with our doctors."
        return
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": prompt})
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "stream": True,
        "max_tokens": 800,
    }
    
    response = make_api_request_with_retry(payload, headers)
    
    if not response:
        yield "Error: Could not connect to AI service. Please try again."
        return

    # Parse streaming response
    try:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error parsing stream: {str(e)}")
        yield f" Error processing response: {str(e)}"



def classify_query(user_message: str):
    """Classify the user's query type."""
    message_lower = user_message.lower()
    
    # Check for image/PDF upload intent
    upload_keywords = ["رفع", "صور", "صورة", "أشعة", "اشعه", "upload", "image", "xray", "x-ray", "pdf", "تقرير", "scan"]
    if any(word in message_lower for word in upload_keywords):
        return "upload"
    
    # Check for appointment booking
    if any(word in message_lower for word in ["book", "appointment", "schedule", "reserve","حجز"]):
        return "appointment"
    
    # Check for availability/doctors
    if any(word in message_lower for word in ["available", "doctor", "when", "slot"]):
        return "availability"
    
    # Check for FAQ
    if any(word in message_lower for word in ["how", "what", "website", "clinic", "hours", "insurance", "cancel", "reschedule", "bring", "secure", "telehealth"]):
        faqs = load_json(FAQ_FILE)
        for faq in faqs:
            if any(keyword in message_lower for keyword in faq["question"].lower().split()):
                return "faq"
    
    # Default to health question
    return "health"


def handle_health_question(user_message: str, history: list = None, target_language: str = 'ar'):
    """Handle general health questions using LLM."""
    language_policy = get_language_policy(target_language)

    system_message = """You are a helpful healthcare information assistant. You provide general health information and answer questions about diseases, symptoms, and wellness.

RESPONSE FORMAT:
- Respond in the SAME LANGUAGE as the user (Arabic or English).
- First, give a BRIEF answer (2-3 sentences max) that directly addresses the question.
- Then suggest a RELEVANT follow-up question about the topic (not a fixed question).
  - The follow-up should be specific to what was discussed.
  - Examples:
    - If discussing weight loss drinks: "هل تريد معرفة أفضل الأوقات لتناول هذه المشروبات؟" or "Would you like to know the best times to drink these?"
    - If discussing headaches: "هل تعاني من أي أعراض أخرى مصاحبة؟" or "Do you have any other accompanying symptoms?"
    - If discussing diabetes: "هل تريد معرفة الأطعمة التي يجب تجنبها؟" or "Would you like to know which foods to avoid?"
- If the user wants more info or says yes, THEN provide a comprehensive answer.
- If they ask a new question, give a brief answer to that instead.

IMPORTANT:
- You are NOT a doctor and cannot provide medical diagnosis or treatment.
- Always recommend consulting with a healthcare professional for specific medical concerns.
- Use numbered lists (1., 2., 3., etc) for symptoms/steps in detailed answers.
- Put each numbered item on a separate line.
- Avoid markdown formatting like **, -, or | characters. Use plain text only.
- You can reference previous messages in our conversation.
- DO NOT use the same follow-up question every time - make it relevant to the topic!"""

    system_message += f"\n\n{language_policy}"
    
    response = get_llm_response(user_message, system_message, history)
    return response


def handle_appointment_request(user_message: str, history: list = None, target_language: str = 'ar'):
    """Handle appointment booking requests."""
    doctors = load_json(DOCTORS_FILE)
    
    doctor_list = "\n".join([f"- {doc['name']} ({doc['specialty']})" for doc in doctors])
    
    if normalize_language_code(target_language) == 'en':
        prompt = f"""The user wants to book an appointment. Respond in English only. Here are available doctors:
{doctor_list}

User message: {user_message}

Please ask which doctor they prefer and what day/time they'd like to schedule.
Do not include Arabic text."""
    else:
        prompt = f"""المستخدم يريد حجز موعد. أجب بالعربية فقط. قائمة الأطباء المتاحين:
{doctor_list}

رسالة المستخدم: {user_message}

اطلب منه اختيار الطبيب واليوم والوقت المناسب.
لا تضف أي نص إنجليزي."""
    
    response = get_llm_response(prompt, None, history)
    return response


def handle_availability_query(user_message: str, history: list = None, target_language: str = 'ar'):
    """Handle doctor availability queries."""
    doctors = load_json(DOCTORS_FILE)
    
    doctor_info = json.dumps(doctors, indent=2)
    
    if normalize_language_code(target_language) == 'en':
        system_message = f"""You are a healthcare scheduling assistant. Respond in English only. Here is the doctor availability information:
{doctor_info}

Answer the user's questions about doctor availability, specialties, and scheduling.
Do not include Arabic text."""
    else:
        system_message = f"""أنت مساعد مواعيد طبية. أجب بالعربية فقط. معلومات توافر الأطباء:
{doctor_info}

أجب على أسئلة المستخدم حول التوافر والتخصصات والمواعيد.
لا تضف أي نص إنجليزي."""
    
    response = get_llm_response(user_message, system_message, history)
    return response


def handle_faq(user_message: str, history: list = None, target_language: str = 'ar'):
    """Handle FAQ queries."""
    faqs = load_json(FAQ_FILE)
    
    faq_text = "\n\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faqs])
    
    if normalize_language_code(target_language) == 'en':
        system_message = f"""You are a healthcare clinic assistant. Respond in English only. Here are frequently asked questions and answers:

{faq_text}

Answer the user's question based on the FAQ information provided.
Do not include Arabic text."""
    else:
        system_message = f"""أنت مساعد عيادة طبية. أجب بالعربية فقط. فيما يلي الأسئلة الشائعة وإجاباتها:

{faq_text}

أجب عن سؤال المستخدم بناءً على معلومات FAQ.
لا تضف أي نص إنجليزي."""
    
    response = get_llm_response(user_message, system_message, history)
    return response


@app.route('/')
def index():
    """Serve the main chatbot page."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with conversation memory per patient (Streaming)."""
    data = request.json or {}
    user_message = data.get('message', '').strip()
    patient_id = data.get('patient_id', 'anonymous')  # Default to 'anonymous' if not provided
    preferred_language = normalize_language_code(data.get('language_preference', 'ar'))
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    # Get recent history from persistent storage (last 20 messages).
    history = get_chat_messages(patient_id, limit=HISTORY_WINDOW_MESSAGES)
    
    # Classify the query
    query_type = classify_query(user_message)
    response_language = resolve_response_language(user_message, preferred_language)
    
    def generate():
        # Route to appropriate handler with history
        if query_type == "upload":
            # For upload prompt, we just return the string immediately
            gen_response = build_upload_guidance_message(response_language)
        elif query_type == "health":
            gen_response = handle_health_question(user_message, history, target_language=response_language)
        elif query_type == "appointment":
            gen_response = handle_appointment_request(user_message, history, target_language=response_language)
        elif query_type == "availability":
            gen_response = handle_availability_query(user_message, history, target_language=response_language)
        elif query_type == "faq":
            gen_response = handle_faq(user_message, history, target_language=response_language)
        else:
            if response_language == 'en':
                gen_response = "I'm not sure how to help with that. Try asking about symptoms, booking an appointment, or general clinic information."
            else:
                gen_response = "غير متأكد كيف أساعدك في ذلك الآن. يمكنك السؤال عن الأعراض، أو حجز موعد، أو معلومات العيادة العامة."
        
        full_response_text = ""
        
        # Stream the response
        try:
            if isinstance(gen_response, str):
                full_response_text = gen_response
                # Yield single chunk for static text
                yield json.dumps({"chunk": gen_response}) + "\n"
            else:
                # Iterate generator for streaming content
                for chunk in gen_response:
                    full_response_text += chunk
                    yield json.dumps({"chunk": chunk}) + "\n"
        except Exception as e:
            error_chunk = f"\nError generating response: {str(e)}"
            full_response_text += error_chunk
            yield json.dumps({"chunk": error_chunk}) + "\n"
        
        # Store the exchange in persistent history.
        save_chat_message(patient_id, "user", user_message)
        save_chat_message(patient_id, "assistant", full_response_text)
        trim_chat_messages(patient_id, MAX_HISTORY_MESSAGES)
        
        # Final detailed JSON with done=true
        yield json.dumps({
            "done": True,
            "response": full_response_text,
            "query_type": query_type,
            "patient_id": patient_id,
            "show_upload": query_type == "upload"
        }) + "\n"

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')


@app.route('/api/doctors', methods=['GET'])
def get_doctors():
    """Get list of all doctors."""
    doctors = load_json(DOCTORS_FILE)
    return jsonify(doctors)


@app.route('/api/appointments/book', methods=['POST'])
def book_appointment():
    """Book an appointment."""
    data = request.json
    
    appointment = {
        "id": len(load_json(APPOINTMENTS_FILE)) + 1,
        "patient_id": data.get('patient_id'),  # Link to patient
        "patient_name": data.get('patient_name'),
        "doctor_id": data.get('doctor_id'),
        "date": data.get('date'),
        "time": data.get('time'),
        "booked_at": datetime.now().isoformat()
    }
    
    appointments = load_json(APPOINTMENTS_FILE)
    appointments.append(appointment)
    save_json(APPOINTMENTS_FILE, appointments)
    
    return jsonify({
        "success": True,
        "message": "Appointment booked successfully",
        "appointment": appointment
    })


@app.route('/api/appointments/queue', methods=['GET'])
def get_queue():
    """Get queue information for a specific doctor and time."""
    doctor_id = request.args.get('doctor_id', type=int)
    date = request.args.get('date')
    time = request.args.get('time')
    
    appointments = load_json(APPOINTMENTS_FILE)
    
    queue_count = len([
        apt for apt in appointments
        if apt['doctor_id'] == doctor_id and apt['date'] == date and apt['time'] == time
    ])
    
    return jsonify({
        "queue_position": queue_count + 1,
        "people_before_you": queue_count
    })


@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear conversation history for a patient."""
    data = request.json
    patient_id = data.get('patient_id', 'anonymous')

    clear_chat_messages(patient_id)
    
    return jsonify({
        "success": True,
        "message": "Conversation history cleared",
        "patient_id": patient_id
    })


@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get conversation history for a patient."""
    patient_id = request.args.get('patient_id', 'anonymous')

    history = get_chat_messages(patient_id)
    
    return jsonify({
        "patient_id": patient_id,
        "history": history,
        "message_count": len(history)
    })


@app.route('/api/patient/<patient_id>/appointments', methods=['GET'])
def get_patient_appointments(patient_id):
    """Get appointments for a specific patient."""
    appointments = load_json(APPOINTMENTS_FILE)
    
    patient_appointments = [
        apt for apt in appointments
        if apt.get('patient_id') == patient_id
    ]
    
    return jsonify({
        "patient_id": patient_id,
        "appointments": patient_appointments,
        "count": len(patient_appointments)
    })


# ==================== FILE UPLOAD ENDPOINTS ====================

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def analyze_image_with_api(image_bytes, image_type):
    """Send image to appropriate medical API for analysis."""
    try:
        if image_type == 'bone':
            api_url = f"{BONE_DETECT_API}/predict_for_llm"
        elif image_type == 'dental_photo':
            # Intraoral photos -> ConvNeXt classifier with GradCAM-derived bbox
            api_url = f"{ORAL_CLASSIFY_API}/predict_for_llm"
        elif image_type in ['dental', 'dental_xray']:
            # Panoramic dental X-rays -> YOLO detection
            api_url = f"{ORAL_DETECT_API}/predict_for_llm"
        elif image_type == 'chest':
            api_url = f"{CHEST_XRAY_API}/predict_for_llm?include_gradcam=true"
        elif image_type == 'prescription':
            # Handwritten prescription -> Qwen2-VL VLM + RapidFuzz against Egyptian drugs DB
            api_url = f"{PRESCRIPTION_API}/predict_for_llm"
        else:
            return {"error": f"Unknown image type: {image_type}"}

        
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        # Prescriptions can require lazy-loading a VLM (~minutes on first call).
        upload_timeout = 1800 if image_type == 'prescription' else 180
        response = requests.post(api_url, files=files, timeout=upload_timeout)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to {image_type} API. Make sure it's running."}
    except Exception as e:
        return {"error": str(e)}


def parse_detections_for_overlay(analysis_result):
    """Extract bbox detections from API results for visualization."""
    detections = []

    raw_detections = analysis_result.get("detections")
    if isinstance(raw_detections, list):
        for det in raw_detections:
            bbox = det.get("bbox") if isinstance(det, dict) else None
            if isinstance(bbox, list) and len(bbox) >= 4:
                detections.append({
                    "bbox": bbox[:4],
                    "label": det.get("class_name") or det.get("finding") or "Finding",
                    "confidence": det.get("confidence"),
                })

    if detections:
        return detections

    raw_findings = analysis_result.get("ai_findings")
    if isinstance(raw_findings, list):
        for finding in raw_findings:
            if not isinstance(finding, dict):
                continue
            location = str(finding.get("location", ""))
            if "bbox" not in location:
                continue
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", location)]
            if len(nums) >= 4:
                detections.append({
                    "bbox": nums[:4],
                    "label": finding.get("finding") or "Finding",
                    "confidence": finding.get("confidence"),
                })

    return detections


def prepare_detections_for_overlay(detections, image_type):
    """Clean and cap detections to keep output overlays readable."""
    if not isinstance(detections, list):
        return []

    def confidence_to_score(value):
        try:
            if value is None:
                return -1.0
            if isinstance(value, str):
                numeric = float(value.strip().replace('%', ''))
                return numeric / 100.0 if numeric > 1.0 else numeric
            numeric = float(value)
            return numeric / 100.0 if numeric > 1.0 else numeric
        except (TypeError, ValueError):
            return -1.0

    cleaned = []
    for det in detections:
        if not isinstance(det, dict):
            continue

        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        cleaned.append({
            "bbox": [x1, y1, x2, y2],
            "label": det.get("label", "Finding"),
            "confidence": det.get("confidence"),
            "score": confidence_to_score(det.get("confidence")),
        })

    cleaned.sort(key=lambda item: item["score"], reverse=True)
    limit = OVERLAY_MAX_DETECTIONS_BY_IMAGE_TYPE.get(image_type, OVERLAY_DEFAULT_MAX_DETECTIONS)
    if limit > 0:
        cleaned = cleaned[:limit]

    return cleaned


def draw_detections_on_image(image_bytes, detections, image_type='dental'):
    """Draw simplified detection markers on the image and return base64 JPEG."""
    if not IMAGE_SUPPORT or not detections:
        return None

    try:
        detections = prepare_detections_for_overlay(detections, image_type)
        if not detections:
            return None

        # Open image safely. PIL converting 16-bit grayscale X-rays directly to
        # RGB can produce a fully-white image due to precision loss, so we
        # normalize high-bitdepth modes first.
        raw = Image.open(io.BytesIO(image_bytes))
        if raw.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
            arr = np.array(raw).astype(np.float32)
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
            image = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        else:
            image = raw.convert("RGB")
        draw = ImageDraw.Draw(image)

        img_w, img_h = image.size
        min_side = max(1, min(img_w, img_h))
        line_w = max(2, int(min_side * 0.003))

        try:
            from PIL import ImageFont
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                max(14, int(min_side * 0.022)),
            )
        except Exception:
            font = None

        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0.0, min(float(x1), img_w - 1))
            y1 = max(0.0, min(float(y1), img_h - 1))
            x2 = max(0.0, min(float(x2), img_w - 1))
            y2 = max(0.0, min(float(y2), img_h - 1))

            color = (255, 80, 80)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

            label = str(det.get("label", "Finding"))
            conf = det.get("confidence")
            try:
                cv = float(conf)
                if cv <= 1.0:
                    cv *= 100.0
                label = f"{label} {cv:.0f}%"
            except (TypeError, ValueError):
                pass

            if font is not None:
                try:
                    tb = draw.textbbox((0, 0), label, font=font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                except Exception:
                    tw, th = (len(label) * 8, 14)
                pad = 3
                ty = max(0, y1 - th - 2 * pad)
                draw.rectangle([x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad], fill=color)
                draw.text((x1 + pad, ty + pad), label, fill=(255, 255, 255), font=font)

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG", quality=92)
        img_bytes.seek(0)
        return base64.b64encode(img_bytes.read()).decode()
    except Exception as e:
        print(f"⚠️ Error drawing detections: {e}")
        return None


def fetch_detection_overlay_detections(image_bytes, image_type):
    """Fetch bbox detections from detection APIs for overlay if missing."""
    if image_type == 'bone':
        api_url = f"{BONE_DETECT_API}/predict"
    elif image_type == 'dental_photo':
        api_url = f"{ORAL_CLASSIFY_API}/predict_for_llm"
    elif image_type in ['dental', 'dental_xray']:
        api_url = f"{ORAL_DETECT_API}/predict"
    else:
        return []

    try:
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(api_url, files=files, timeout=60)
        if response.status_code != 200:
            return []
        data = response.json()
        detections = data.get("detections")
        if isinstance(detections, list):
            return [
                {
                    "bbox": det.get("bbox"),
                    "label": det.get("class_name") or det.get("finding") or "Finding",
                    "confidence": det.get("confidence"),
                }
                for det in detections
                if isinstance(det, dict)
            ]
    except Exception:
        return []

    return []


def extract_pdf_text(pdf_bytes):
    """Extract text from PDF file."""
    if not PDF_SUPPORT:
        return {"error": "PDF support not available. Install pdfplumber."}
    
    try:
        text_content = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- صفحة {i+1} ---\n{page_text}")
        
        if text_content:
            return {"success": True, "text": "\n\n".join(text_content), "pages": len(text_content)}
        else:
            return {"error": "لم يتم العثور على نص في الملف"}
    except Exception as e:
        return {"error": f"خطأ في قراءة PDF: {str(e)}"}


def confidence_to_percent(value):
    """Convert confidence values to percentage (0-100)."""
    try:
        if isinstance(value, str):
            numeric = float(value.strip().replace('%', ''))
        else:
            numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric <= 1.0:
        numeric *= 100.0

    return max(0.0, min(100.0, numeric))


def normalize_oral_label_ar(label):
    """Normalize technical oral labels to patient-friendly Arabic terms."""
    text = str(label or '').strip().lower().replace('_', ' ').replace('-', ' ')
    text = re.sub(r'\s+', ' ', text)

    if 'ulcer' in text:
        return 'قرحة فموية'
    if 'gingiv' in text:
        return 'التهاب اللثة'
    if 'calculus' in text or 'tartar' in text:
        return 'جير على الأسنان'
    if 'hypodontia' in text or 'missing tooth' in text or 'missing' in text:
        return 'سن مفقود'
    if 'wisdom' in text:
        return 'ضرس العقل'
    if 'apical periodontitis' in text or 'periodontitis' in text:
        return 'التهاب لب جذر السن'
    if 'root canal' in text:
        return 'حشوة عصب (علاج جذر)'
    if 'porcelain crown' in text or ('crown' in text and 'porcelain' in text):
        return 'تاج خزفي'
    if 'crown' in text:
        return 'تاج سن'
    if 'ceramic bridge' in text or 'bridge' in text:
        return 'جسر سني'
    if 'implant' in text:
        return 'زرعة سن'
    if 'dental filling' in text or 'filling' in text:
        return 'حشوة أسنان'
    if 'discolor' in text or 'stain' in text:
        return 'تغير لون الأسنان'
    if 'caries' in text or 'decay' in text or 'cavity' in text:
        return 'تسوس أسنان'
    if re.search(r'class\s*\d+', text):
        return 'تسوس أسنان'

    return 'مؤشر فموي يحتاج تقييم'


def normalize_chest_label_ar(label):
    """Translate chest class labels to Arabic-friendly names."""
    mapping = {
        'Atelectasis': 'انخماص الرئة',
        'Cardiomegaly': 'تضخم القلب',
        'Consolidation': 'تصلب رئوي',
        'Edema': 'وذمة رئوية',
        'Effusion': 'انصباب جنبي',
        'Emphysema': 'انتفاخ الرئة',
        'Fibrosis': 'تليف رئوي',
        'Infiltration': 'ارتشاح رئوي',
        'Mass': 'كتلة رئوية',
        'Nodule': 'عقيدة رئوية',
        'Pleural_Thickening': 'سماكة غشاء الجنب',
        'Pneumonia': 'التهاب رئوي',
        'Pneumothorax': 'استرواح صدري',
    }
    return mapping.get(str(label or '').strip(), str(label or 'مؤشر صدري'))


def _bbox_iou(a, b):
    try:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
    except Exception:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedup_detections_by_iou(detections, iou_threshold=0.4):
    """Greedy NMS-style dedup: keep highest-conf box per class cluster.

    detections: list of {'class', 'conf', 'bbox': [x1,y1,x2,y2]}.
    """
    sorted_dets = sorted(detections, key=lambda d: d.get('conf', 0), reverse=True)
    kept = []
    for d in sorted_dets:
        duplicate = False
        for k in kept:
            if k['class'] != d['class']:
                continue
            if _bbox_iou(k.get('bbox') or [0, 0, 0, 0], d.get('bbox') or [0, 0, 0, 0]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    return kept


# Per-finding clinical recommendations (Arabic) keyed by canonical English label (lowercased).
PER_FINDING_RECOMMENDATIONS_AR = {
    # ---- Oral X-ray (9-class) ----
    "wisdom tooth": "متابعة دورية للأسنان العقل؛ في حال كانت مدفونة أو تسبب التهاب أو ازدحام يُنصح بالخلع الجراحي بعد تقييم طبيب الأسنان.",
    "missing tooth": "النظر في خيار زرعة أو جسر لتعويض الفراغ ومنع تحرك الأسنان المجاورة وفقدان عظم الفك.",
    "decay": "حشو السن المتسوس بعد إزالة النخر؛ في الحالات المتقدمة قد يلزم علاج عصب أو تركيب تاج.",
    "caries": "حشو السن المتسوس بعد إزالة النخر؛ في الحالات المتقدمة قد يلزم علاج عصب أو تركيب تاج.",
    "porcelain crown": "تاج موجود — متابعة دورية للتأكد من سلامة التاج واللثة المحيطة وعدم وجود تسوس تحت التاج.",
    "crown": "تاج موجود — متابعة دورية للتأكد من سلامة التاج واللثة المحيطة وعدم وجود تسوس تحت التاج.",
    "ceramic bridge": "جسر سيراميك موجود — متابعة دورية ونظافة جيدة للأسنان الداعمة للجسر.",
    "bridge": "جسر سني موجود — متابعة دورية ونظافة جيدة للأسنان الداعمة.",
    "implant": "زرعة سنية موجودة — متابعة دورية للتأكد من ثبات الزرعة وصحة العظم واللثة المحيطة.",
    "dental filling": "حشوة موجودة — متابعة دورية للتأكد من سلامة الحشوة وعدم وجود تسوس متكرر حولها.",
    "filling": "حشوة موجودة — متابعة دورية للتأكد من سلامة الحشوة وعدم وجود تسوس متكرر حولها.",
    "root canal filling": "علاج عصب موجود — متابعة دورية بالأشعة للتأكد من نجاح العلاج وعدم وجود التهاب جذري.",
    "root canal": "علاج عصب موجود — متابعة دورية بالأشعة للتأكد من نجاح العلاج وعدم وجود التهاب جذري.",
    "apical periodontitis": "التهاب حول قمي يحتاج علاج عصب الجذر؛ في الحالات المتقدمة قد يلزم استئصال القمة أو خلع السن.",
    # ---- Oral photo (6-class intraoral) ----
    "calculus": "تنظيف وتلميع الأسنان لدى طبيب الأسنان (سكلينج) مع تعليمات نظافة فموية يومية واستخدام خيط الأسنان.",
    "gingivitis": "تنظيف لثوي عميق ومضمضة بمحلول مطهر (مثل الكلورهيكسيدين) لمدة محددة؛ تحسين النظافة الفموية لمنع التطور لالتهاب لثة مزمن.",
    "ulcer": "علاج موضعي مهدئ (جل أو غسول)؛ تجنب الأطعمة الحارة؛ مراجعة الطبيب إذا استمرت القرحة أكثر من أسبوعين.",
    "mouth ulcer": "علاج موضعي مهدئ (جل أو غسول)؛ تجنب الأطعمة الحارة؛ مراجعة الطبيب إذا استمرت القرحة أكثر من أسبوعين.",
    "tooth discoloration": "تحديد سبب التغير (داخلي أو خارجي)؛ تنظيف احترافي أو تبييض أسنان حسب الحالة.",
    "discoloration": "تحديد سبب التغير (داخلي أو خارجي)؛ تنظيف احترافي أو تبييض أسنان حسب الحالة.",
    "hypodontia": "استشارة أخصائي تقويم وتعويضات سنية لتقييم خيارات تعويض الأسنان المفقودة خلقياً.",
    # ---- Bone X-ray ----
    "fracture": "تثبيت الكسر بالجبيرة أو الجبس أو التدخل الجراحي حسب نوع الكسر؛ متابعة بالأشعة للتأكد من الالتئام السليم.",
    "boneanomaly": "تقييم طبي تفصيلي للشذوذ العظمي مع أشعة إضافية إذا لزم الأمر.",
    "bonelesion": "تقييم آفة العظم بأشعة مقطعية أو رنين مغناطيسي وقد يلزم خزعة لتحديد الطبيعة.",
    "foreignbody": "تقييم لإزالة الجسم الغريب جراحياً إذا كان يسبب أعراضاً أو خطر عدوى.",
    "metal": "تثبيت معدني موجود — متابعة الالتئام وتقييم وجود التهاب أو ارتخاء.",
    "periostealreaction": "متابعة الالتئام وتقييم وجود عدوى أو ورم؛ قد يلزم تصوير إضافي.",
    "pronatorsign": "علامة على كسر دقيق محتمل في الساعد — تقييم سريري وأشعة إضافية.",
    "softtissue": "تقييم تورم الأنسجة الرخوة لاحتمال وجود التهاب أو نزيف داخلي.",
    "text": "",
    # ---- Chest X-ray ----
    "pneumonia": "علاج بالمضادات الحيوية حسب نوع العدوى؛ راحة وترطيب جيد؛ متابعة بأشعة بعد 4–6 أسابيع للتأكد من الشفاء.",
    "tuberculosis": "علاج طويل الأمد بأدوية السل المركبة (6 أشهر على الأقل)؛ متابعة دورية وعزل حسب الإرشادات.",
    "covid-19": "عزل وراحة ومتابعة الأكسجين؛ علاج عرضي وأدوية مضادة للفيروسات في الحالات المتوسطة والشديدة.",
    "covid": "عزل وراحة ومتابعة الأكسجين؛ علاج عرضي وأدوية مضادة للفيروسات في الحالات المتوسطة والشديدة.",
    "lung opacity": "تقييم إضافي لتحديد سبب العتامة (التهاب، وذمة، أو ورم)؛ قد يلزم أشعة مقطعية.",
    "viral pneumonia": "علاج عرضي وراحة وترطيب؛ مضادات فيروسية حسب نوع الفيروس؛ متابعة طبية.",
    "bacterial pneumonia": "علاج بمضادات حيوية مناسبة؛ متابعة بأشعة بعد العلاج للتأكد من الشفاء.",
    "edema": "علاج السبب الأساسي (قصور قلب، فشل كلوي)؛ مدرات بول ومتابعة وظائف القلب.",
    "effusion": "بزل السائل من الجنب لتحليله إذا كان كبيراً؛ علاج السبب الأساسي.",
    "pleural effusion": "بزل السائل من الجنب لتحليله إذا كان كبيراً؛ علاج السبب الأساسي.",
    "consolidation": "غالباً علامة على التهاب رئوي — علاج بالمضادات الحيوية ومتابعة دورية بالأشعة.",
    "infiltration": "تقييم لتحديد السبب (التهاب، عدوى، حساسية)؛ متابعة بأشعة لاحقة.",
    "cardiomegaly": "تقييم وظائف القلب بإيكو وتخطيط قلب؛ علاج السبب (ارتفاع ضغط، اعتلال عضلة قلب).",
    "atelectasis": "تنفس عميق وتمارين تنفسية؛ علاج السبب الأساسي (انسداد قصبي، ضغط من خارج الرئة).",
    "pneumothorax": "إذا كان كبيراً يتطلب تدخل عاجل لتركيب أنبوب صدري لتفريغ الهواء؛ متابعة دقيقة.",
    "mass": "أشعة مقطعية بالصبغة وقد يلزم خزعة لتحديد طبيعة الكتلة (حميدة/خبيثة).",
    "nodule": "متابعة بأشعة مقطعية كل 3–6 أشهر لتقييم النمو؛ خزعة إذا كانت العقدة مشبوهة.",
    "fibrosis": "تقييم بأشعة مقطعية ووظائف رئة؛ علاج داعم لإبطاء التطور.",
    "emphysema": "إيقاف التدخين فوراً، علاج بموسعات قصبية واستنشاقات، تأهيل رئوي.",
    "pleural thickening": "متابعة دورية بالأشعة؛ تقييم تاريخ التعرض للأسبستوس أو الالتهابات السابقة.",
    "hernia": "تقييم جراحي لاحتمال الإصلاح حسب الحجم والأعراض.",
    "no finding": "لا توجد نتائج مرضية واضحة — استمر في المتابعة الدورية والوقاية الصحية.",
    "normal": "النتائج طبيعية — استمر في المتابعة الدورية والوقاية الصحية.",
}


def _normalize_label(label):
    if not label:
        return ""
    return str(label).strip().lower()


def build_per_finding_recommendations(analysis_result, image_type=None):
    """Return an Arabic block of per-finding clinical recommendations.

    Collects unique findings from `detections[*].class_name` and
    `ai_findings.primary_diagnosis` / `differential_diagnoses`, then maps each
    to a curated Arabic treatment recommendation from PER_FINDING_RECOMMENDATIONS_AR.
    """
    if not isinstance(analysis_result, dict):
        return ""

    seen = set()
    items = []  # list of (display_label, recommendation)

    def _add(label):
        key = _normalize_label(label)
        if not key or key in seen:
            return
        rec = PER_FINDING_RECOMMENDATIONS_AR.get(key)
        if not rec:
            return
        seen.add(key)
        items.append((label, rec))

    for det in analysis_result.get("detections", []) or []:
        if isinstance(det, dict):
            _add(det.get("class_name") or det.get("label") or det.get("class"))

    ai = analysis_result.get("ai_findings") or {}
    if isinstance(ai, dict):
        _add(ai.get("primary_diagnosis"))
        for d in ai.get("differential_diagnoses") or []:
            if isinstance(d, dict):
                _add(d.get("condition") or d.get("diagnosis") or d.get("label"))
            else:
                _add(d)

    if not items:
        return ""

    lines = ["", "📋 توصيات علاجية مقترحة لكل حالة (مرجع للطبيب):"]
    for label, rec in items:
        lines.append(f"- {label}: {rec}")
    return "\n".join(lines)


def build_prescription_report(analysis_result):
    """Build a structured Arabic report for the prescription parser output.

    Skips the LLM since the VLM already produced verified structured data.
    """
    if not isinstance(analysis_result, dict):
        return "تعذّر قراءة الروشتة."

    ai = analysis_result.get('ai_findings') or {}
    meds = ai.get('medications') or []
    if not meds and isinstance(analysis_result.get('report_data'), dict):
        meds = analysis_result['report_data'].get('medications') or []

    lines = ["📋 تحليل الروشتة (Handwritten Prescription):"]
    if not meds:
        lines.append("لم يتم استخراج أي أدوية واضحة من الصورة.")
        lines.append("")
        lines.append("⏰ التوصية: حاول رفع صورة أوضح للروشتة أو تواصل مع الصيدلي.")
        return "\n".join(lines)

    def _clean_text(v):
        return re.sub(r"\s+", " ", str(v or "").strip())

    def _alpha_len(v):
        return len(re.findall(r"[A-Za-z\u0621-\u064A]", _clean_text(v)))

    def _is_weak_name(v):
        t = _clean_text(v).lower()
        if not t or t in {"-", "—", "n/a", "na", "none", "null"}:
            return True
        return _alpha_len(t) <= 2

    def _simple_schedule_ar(freq_raw, dosage_raw):
        freq = _clean_text(freq_raw).lower()
        dosage = _clean_text(dosage_raw).lower()

        if not freq:
            return "غير واضح"
        if re.search(r"عند\s*اللزو?م|عند\s*الحاجة|\bprn\b", freq):
            return "عند اللزوم"
        m = re.search(r"(?:كل\s*)(\d{1,2})\s*(?:ساع|ساعة|ساعات)|(?:\bq\s*)(\d{1,2})\s*h", freq)
        if m:
            h = m.group(1) or m.group(2)
            return f"كل {h} ساعات"

        count = ""
        if re.search(r"\b(bid|مرتين|twice)\b", freq):
            count = "مرتين يوميا"
        elif re.search(r"\b(tid|ثلاث|3\s*مرات|three times)\b", freq):
            count = "3 مرات يوميا"
        elif re.search(r"\b(qid|اربع|4\s*مرات|four times)\b", freq):
            count = "4 مرات يوميا"
        elif re.search(r"\b(od|qd|once|daily|يومي|مرة)\b", freq) or freq in {"1", "1x", "1 tab", "1 cap", "1 dose", "1 amp", "1 drop"}:
            count = "مرة يوميا"
        elif freq in {"2", "2x", "2 tab", "2 cap", "2 dose"}:
            count = "مرتين يوميا"
        elif freq in {"3", "3x", "3 tab", "3 cap", "3 dose"}:
            count = "3 مرات يوميا"

        parts = []
        if count:
            parts.append(count)

        if re.search(r"قبل\s*ال[اأ]?كل|before\s*meal|before\s*food", freq):
            parts.append("قبل الأكل")
        elif re.search(r"بعد\s*ال[اأ]?كل|after\s*meal|after\s*food", freq):
            parts.append("بعد الأكل")

        if re.search(r"قبل\s*النوم|before\s*sleep|bedtime|\bhs\b", freq):
            parts.append("قبل النوم")

        times = []
        if re.search(r"صباح|صبح|morning|\bam\b|الفطار", freq):
            times.append("صباحا")
        if re.search(r"ظهر|noon|الغدا", freq):
            times.append("ظهرا")
        if re.search(r"مساء|ليل|عشاء|bedtime|\bpm\b|\bhs\b|evening|night", freq):
            times.append("مساء")
        if times:
            parts.append(" و ".join(times))

        schedule = " - ".join(parts).strip(" -")
        if schedule == "مرة يوميا" and any(k in dosage for k in ("amp", "ampoule", "vial", "inj", "حقن")):
            return "جرعة واحدة"
        return schedule or "غير واضح"

    def _display_name_and_flag(med):
        official = _clean_text(med.get('drug'))
        extracted = _clean_text(med.get('drug_extracted'))
        score = float(med.get('confidence_score') or 0)
        official_match = bool(med.get('official_match'))

        official_weak = _is_weak_name(official)
        extracted_weak = _is_weak_name(extracted)

        # Avoid showing short/noisy tokens as final names.
        if official_weak and not extracted_weak:
            return extracted, "⚠️ يحتاج مراجعة صيدلي"
        if official_match and score >= 92 and not official_weak:
            return official, "✅ مطابق قوي"
        if official_match and score >= 82 and not official_weak:
            return official, "🟡 مطابق تقريبي"
        if not extracted_weak:
            return extracted, "⚠️ يحتاج مراجعة صيدلي"
        if not official_weak:
            return official, "⚠️ يحتاج مراجعة صيدلي"
        return "اسم غير واضح", "⚠️ يحتاج مراجعة صيدلي"

    verified_strong = 0

    lines.append(f"تم رصد {len(meds)} دواء من الروشتة.")
    lines.append("")
    lines.append("💊 الأدوية المكتشفة (نسخة مبسطة):")
    for i, med in enumerate(meds, 1):
        name, name_flag = _display_name_and_flag(med)

        score = float(med.get('confidence_score') or 0)
        if name_flag.startswith("✅"):
            verified_strong += 1

        lines.append(f"{i}) {name} ({name_flag})")
        if 0 < score < 92:
            lines.append(f"   - درجة المطابقة: {score:.0f}%")

    lines.append("")
    lines.append(f"✅ أسماء مطابقة بقوة: {verified_strong} من {len(meds)}")
    lines.append("⏰ المتابعة: لأي تفاصيل جرعة/توقيت، الرجاء الرجوع مباشرة إلى نص الروشتة أو الصيدلي.")
    return "\n".join(lines)


def generate_medical_report(analysis_result, file_type, image_type=None, target_language='ar'):
    """Generate medical report using LLM based on analysis results."""
    response_language = normalize_language_code(target_language)

    if not LLM_AVAILABLE:
        if response_language == 'en':
            return f"Analysis results:\n{json.dumps(analysis_result, ensure_ascii=False, indent=2)}"
        return f"نتائج التحليل:\n{json.dumps(analysis_result, ensure_ascii=False, indent=2)}"
    
    if file_type == 'image':
        if 'error' in analysis_result:
            if response_language == 'en':
                return f"Analysis error: {analysis_result['error']}"
            return f"خطأ في التحليل: {analysis_result['error']}"

        # ---- Prescription parsing has its own structured report (no LLM needed) ----
        if image_type == 'prescription':
            prescription_report = build_prescription_report(analysis_result)
            if response_language == 'en':
                translation_prompt = f"""Translate the following medical report from Arabic to clear English.
Rules:
- Preserve the original structure, numbering, and emojis.
- Keep medication names as they are when possible.
- Keep safety warnings explicit.

Report:
{prescription_report}
"""
                translated = resolve_llm_response(
                    get_llm_response(
                        translation_prompt,
                        "You are a medical translator. Return only the translated report in English.",
                    )
                )
                return clean_markdown(translated)
            return prescription_report

        findings_list = []

        if image_type in ['dental', 'dental_photo', 'dental_xray']:
            # Group detections by class with NMS dedup so each tooth/lesion is
            # counted once. Then list count + confidence range per class.
            raw = []
            for d in analysis_result.get('detections', []):
                if not isinstance(d, dict):
                    continue
                class_name = d.get('class_name')
                if not class_name:
                    continue
                conf_pct = confidence_to_percent(d.get('confidence'))
                if conf_pct is None or conf_pct < 30:
                    continue
                bbox = d.get('bbox') or [0, 0, 0, 0]
                raw.append({'class': class_name, 'conf': conf_pct, 'bbox': bbox})
            kept = dedup_detections_by_iou(raw, iou_threshold=0.4)
            grouped = {}
            for d in kept:
                label_ar = normalize_oral_label_ar(d['class'])
                grouped.setdefault(label_ar, []).append(d['conf'])
            for label_ar, confs in sorted(grouped.items(), key=lambda x: max(x[1]), reverse=True):
                confs_sorted = sorted(confs, reverse=True)
                if len(confs_sorted) == 1:
                    findings_list.append(f"- يرجح وجود {label_ar} (احتمال {confs_sorted[0]:.0f}%)")
                else:
                    confs_str = ", ".join(f"{c:.0f}%" for c in confs_sorted)
                    findings_list.append(
                        f"- يرجح وجود {label_ar} في {len(confs_sorted)} مواضع (احتمالات: {confs_str})"
                    )

            # Classifier-style fallback: intraoral photo classifier returns
            # no detections, only ai_findings + all_probabilities. List the
            # primary diagnosis and any differentials >=20%.
            if not findings_list:
                ai_findings = analysis_result.get('ai_findings')
                if isinstance(ai_findings, dict):
                    primary = ai_findings.get('primary_diagnosis')
                    primary_conf = confidence_to_percent(ai_findings.get('confidence'))
                    if primary:
                        label_ar = normalize_oral_label_ar(primary)
                        if primary_conf is not None:
                            findings_list.append(
                                f"- يرجح وجود {label_ar} (احتمال {primary_conf:.0f}%)"
                            )
                        else:
                            findings_list.append(f"- يرجح وجود {label_ar}")

                all_probs = analysis_result.get('all_probabilities') or {}
                seen_labels = set()
                for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                    pct = confidence_to_percent(prob)
                    if pct is None or pct < 20:
                        continue
                    label_ar = normalize_oral_label_ar(cls)
                    if label_ar in seen_labels:
                        continue
                    # Skip the primary one we already added.
                    primary_ar = normalize_oral_label_ar(
                        (analysis_result.get('ai_findings') or {}).get('primary_diagnosis') or ''
                    )
                    if label_ar == primary_ar:
                        seen_labels.add(label_ar)
                        continue
                    seen_labels.add(label_ar)
                    findings_list.append(f"- احتمال إضافي: {label_ar} ({pct:.0f}%)")

        elif image_type == 'chest':
            ai_findings = analysis_result.get('ai_findings')
            if isinstance(ai_findings, dict):
                primary = ai_findings.get('primary_diagnosis')
                primary_conf = confidence_to_percent(ai_findings.get('confidence'))
                if primary:
                    primary_label = normalize_chest_label_ar(primary)
                    if primary_conf is not None:
                        findings_list.append(f"- التشخيص الأكثر ترجيحًا: {primary_label} (احتمال {primary_conf:.0f}%)")
                    else:
                        findings_list.append(f"- التشخيص الأكثر ترجيحًا: {primary_label}")

            for cond in analysis_result.get('detected_conditions', []):
                if not isinstance(cond, dict):
                    continue
                raw_label = cond.get('condition') or cond.get('class')
                conf_pct = confidence_to_percent(cond.get('probability'))
                if not raw_label or conf_pct is None or conf_pct < 30:
                    continue
                findings_list.append(
                    f"- احتمال إضافي: {normalize_chest_label_ar(raw_label)} ({conf_pct:.0f}%)"
                )

        else:
            # Bone / generic: dedup overlapping detections (YOLO can return
            # near-duplicate boxes for the same fracture) then group per class.
            raw = []
            for d in analysis_result.get('detections', []):
                if not isinstance(d, dict) or not d.get('class_name'):
                    continue
                conf_pct = confidence_to_percent(d.get('confidence'))
                if conf_pct is None or conf_pct < 30:
                    continue
                bbox = d.get('bbox') or [0, 0, 0, 0]
                raw.append({'class': d['class_name'], 'conf': conf_pct, 'bbox': bbox})
            kept = dedup_detections_by_iou(raw, iou_threshold=0.4)
            grouped = {}
            for d in kept:
                grouped.setdefault(d['class'], []).append(d['conf'])
            for cls, confs in sorted(grouped.items(), key=lambda x: max(x[1]), reverse=True):
                confs_sorted = sorted(confs, reverse=True)
                if len(confs_sorted) == 1:
                    findings_list.append(f"- {cls} (احتمال {confs_sorted[0]:.0f}%)")
                else:
                    confs_str = ", ".join(f"{c:.0f}%" for c in confs_sorted)
                    findings_list.append(
                        f"- {cls}: {len(confs_sorted)} مواضع (احتمالات: {confs_str})"
                    )

            ai_findings = analysis_result.get('ai_findings', [])
            if isinstance(ai_findings, list) and not findings_list:
                # Only add ai_findings entries when we don't already have
                # per-detection lines; otherwise we'd duplicate the same
                # finding under two different formats.
                for f in ai_findings:
                    if isinstance(f, dict) and f.get('finding'):
                        findings_list.append(f"- {f['finding']} (ثقة: {f.get('confidence', '؟')})")

        if findings_list:
            findings_intro = "Detected findings:" if response_language == 'en' else "النتائج المكتشفة:"
            findings_text = f"{findings_intro}\n" + "\n".join(findings_list)
        else:
            all_probs = analysis_result.get('all_probabilities', {})
            if all_probs:
                top3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                top3_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in top3)
                if response_language == 'en':
                    findings_text = (
                        "No strong abnormal findings above the configured confidence threshold. "
                        f"Top current probabilities: {top3_str}"
                    )
                else:
                    findings_text = f"لا توجد مؤشرات قوية فوق عتبة الثقة المعتمدة. أعلى الاحتمالات الحالية: {top3_str}"
            else:
                if response_language == 'en':
                    findings_text = "No strong abnormal indicators were detected in the image based on the configured threshold."
                else:
                    findings_text = "لا توجد مؤشرات قوية غير طبيعية في الصورة وفق العتبة المعتمدة."

        # Build per-finding recommendations so the LLM has explicit suggested
        # treatment for each detected condition (instead of generic advice).
        findings_text += build_per_finding_recommendations(analysis_result, image_type)

        if response_language == 'en':
            image_type_label = {
                'bone': 'Bone X-ray (wrist)',
                'chest': 'Chest X-ray',
                'dental': 'Dental image',
                'dental_photo': 'Dental photo',
                'dental_xray': 'Dental X-ray',
                'prescription': 'Handwritten prescription',
            }.get(image_type, 'Medical image')

            prompt = f"""You are a medical specialist. Write a concise English report based on the analysis below.

Image type: {image_type_label}
{findings_text}

Write the report in this exact structure (use emojis):

📋 Detected Findings:
[List each detected condition clearly, or state that no strong abnormal finding is detected]

⚠️ Severity: [low/medium/high]

💊 Recommendations:
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

⏰ Follow-up: [When to seek follow-up]

Important rules:
- Use probabilistic wording only (for example: "suggests", "may indicate"). Do not provide definitive diagnosis.
- Never expose technical class IDs like Class_4 or Class 7 in the patient report.
- In "Detected Findings", preserve each provided finding once without duplication.
- If no strong finding exists, mention that clinical follow-up is recommended if symptoms persist.
- If the input includes "📋 توصيات علاجية مقترحة لكل حالة", use those recommendations as the primary basis for the recommendations section.

Keep it concise and clear. Do not use tables."""
        else:
            image_type_label = {
                'bone': 'أشعة عظام (رسغ)',
                'chest': 'أشعة صدر',
                'dental': 'صورة أسنان',
                'dental_photo': 'صورة أسنان',
                'dental_xray': 'أشعة أسنان',
                'prescription': 'روشتة طبية مكتوبة بخط اليد',
            }.get(image_type, 'صورة طبية')

            prompt = f"""أنت طبيب متخصص. اكتب تقرير طبي مختصر بالعربية بناءً على نتائج التحليل.

نوع الصورة: {image_type_label}
{findings_text}

اكتب التقرير بالشكل التالي (استخدم الرموز):

📋 النتائج المكتشفة:
[اذكر كل حالة مكتشفة بالعربي أو اذكر أن كل شيء طبيعي]

⚠️ الشدة: [منخفضة/متوسطة/عالية]

💊 التوصيات:
1. [توصية 1]
2. [توصية 2]
3. [توصية 3]

⏰ المتابعة: [متى يجب المتابعة]

قواعد مهمة جدًا:
- استخدم صياغة احتمالية فقط مثل: "يرجح"، "قد تشير النتائج"، ولا تكتب تشخيصًا مؤكدًا.
- لا تذكر أي أسماء تقنية مثل Class_4 أو Class 7، واعتبرها ضمن "تسوس أسنان".
- في قسم "النتائج المكتشفة" انسخ كل بند من القائمة المعطاة كما هو بدون تكرار وبدون إضافة بنود جديدة. لا تعيد صياغة نفس البند بأكثر من سطر.
- لو لا توجد نتائج قوية، اذكر أن المتابعة السريرية مطلوبة مع إعادة التقييم عند استمرار الأعراض.
- إذا وُجد قسم "📋 توصيات علاجية مقترحة لكل حالة" ضمن البيانات أعلاه، استخدم هذه التوصيات الموصى بها كأساس لقسم "💊 التوصيات" واكتب توصية محددة لكل حالة مكتشفة (بدلاً من التوصيات العامة)، ويمكنك إضافة توصيات سلوكية عامة في النهاية.

اجعل التقرير مختصر وواضح. لا تستخدم جداول."""

    
    elif file_type == 'pdf':
        if 'error' in analysis_result:
            if response_language == 'en':
                return f"File reading error: {analysis_result['error']}"
            return f"خطأ في قراءة الملف: {analysis_result['error']}"

        if response_language == 'en':
            prompt = f"""You are a medical specialist. Analyze and summarize the following medical report in English:

Report content:
{analysis_result.get('text', '')[:3000]}

Provide a concise summary including:
1. Main findings
2. Any abnormal values
3. Recommendations (if available)"""
        else:
            prompt = f"""أنت طبيب متخصص. قم بتحليل التقرير الطبي التالي وتلخيصه باللغة العربية:

محتوى التقرير:
{analysis_result.get('text', '')[:3000]}

قدم ملخصاً مختصراً يتضمن:
1. النتائج الرئيسية
2. أي قيم غير طبيعية
3. التوصيات إن وجدت"""
    
    else:
        return "نوع ملف غير معروف"
    
    return get_llm_response(prompt)


def resolve_llm_response(report):
    """Ensure LLM response is a plain string for JSON serialization."""
    if isinstance(report, str):
        return report
    try:
        return "".join(chunk for chunk in report)
    except TypeError:
        return str(report)


def generate_followup_for_uploaded_file(user_message, base_report, image_type=None, target_language='ar'):
    """Answer an optional user prompt that is sent with an uploaded file.

    This lets the chatbot return both:
    1) the standard file analysis report, and
    2) a direct reply to the user's attached question.
    """
    if not user_message:
        return ""

    user_message = str(user_message).strip()
    if not user_message:
        return ""

    response_language = resolve_response_language(user_message, target_language)

    if not LLM_AVAILABLE:
        if response_language == 'en':
            return "Your question was received with the file, but the AI service is currently unavailable for a detailed reply."
        return "تم استلام سؤالك مع الملف. الخدمة الذكية غير متاحة حالياً للرد التفصيلي."

    if response_language == 'en':
        image_type_label = {
            'bone': 'Bone X-ray',
            'chest': 'Chest X-ray',
            'dental': 'Dental image/X-ray',
            'dental_photo': 'Dental photo',
            'dental_xray': 'Dental X-ray',
            'prescription': 'Prescription',
            None: 'Medical file',
        }.get(image_type, 'Medical file')

        prompt = f"""You have the following {image_type_label} analysis context:
{base_report}

Attached user question:
{user_message}

Required:
- Answer the question directly and briefly.
- Base your answer only on the provided analysis context.
- If information is uncertain, say so clearly.
- Avoid definitive diagnosis or changing prescribed treatment.
- Reply in the user's language.
"""

        system = (
            "You are a medical explanation assistant. "
            "Provide a concise and safe answer based only on the available analysis, "
            "and do not replace the treating physician's decision."
        )
    else:
        image_type_label = {
            'bone': 'أشعة عظام',
            'chest': 'أشعة صدر',
            'dental': 'صورة/أشعة أسنان',
            'dental_photo': 'صورة أسنان',
            'dental_xray': 'أشعة أسنان',
            'prescription': 'روشتة طبية',
            None: 'ملف طبي',
        }.get(image_type, 'ملف طبي')

        prompt = f"""لديك سياق تحليل {image_type_label} التالي:
{base_report}

سؤال المستخدم المرفق مع الملف:
{user_message}

المطلوب:
- أجب على السؤال مباشرة وباختصار عملي.
- اربط الإجابة بنتيجة التحليل المذكورة أعلاه فقط.
- إذا كانت المعلومة غير مؤكدة من التحليل، اذكر ذلك بوضوح.
- تجنب التشخيص القطعي أو تغيير وصفة الطبيب.
- الرد بنفس لغة المستخدم.
"""

        system = (
            "أنت مساعد طبي توضيحي. "
            "قدم إجابة مختصرة وآمنة بناءً على نتيجة التحليل المتاحة فقط، "
            "ولا تستبدل قرار الطبيب المعالج."
        )

    response = get_llm_response(prompt, system)
    return clean_markdown(resolve_llm_response(response))


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads (images and PDFs) for medical analysis."""
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم إرسال ملف"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "لم يتم اختيار ملف"}), 400
    
    patient_id = request.form.get('patient_id', 'anonymous')
    user_message = request.form.get('user_message', '').strip()
    image_type = request.form.get('image_type', 'dental')  # 'dental' or 'bone'
    language_preference = normalize_language_code(request.form.get('language_preference', 'ar'))
    response_language = resolve_response_language(user_message, language_preference)
    
    filename = file.filename.lower()
    file_bytes = file.read()
    
    # Determine file type and process
    if allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
        if not IMAGE_SUPPORT:
            return jsonify({"error": "دعم الصور غير متاح"}), 500
        
        # Analyze image with appropriate API
        analysis_result = analyze_image_with_api(file_bytes, image_type)

        # Prefer API-provided annotated image (rendered with cv2 on the server,
        # handles 16-bit X-rays correctly and includes class labels + confidences).
        result_image = analysis_result.get('annotated_image_base64') or \
                       analysis_result.get('annotated_image')

        # Fallback: render overlay locally if API did not provide one.
        if not result_image and image_type in {'bone', 'dental', 'dental_photo', 'dental_xray'}:
            detections = parse_detections_for_overlay(analysis_result)
            if not detections:
                detections = fetch_detection_overlay_detections(file_bytes, image_type)
            result_image = draw_detections_on_image(file_bytes, detections, image_type=image_type)
        
        # Generate report
        report = resolve_llm_response(
            generate_medical_report(analysis_result, 'image', image_type, target_language=response_language)
        )
        report = clean_markdown(report)

        # If the user attached a text prompt with the uploaded image,
        # answer it as a second section after the standard analysis report.
        if user_message:
            followup = generate_followup_for_uploaded_file(
                user_message=user_message,
                base_report=report,
                image_type=image_type,
                target_language=response_language,
            )
            if followup:
                followup_heading = "💬 Reply to your message:" if response_language == 'en' else "💬 رد على رسالتك:"
                report = f"{report}\n\n{followup_heading}\n{followup}"
        
        # Store in persistent conversation history
        user_msg = f"[تم رفع صورة أشعة {'عظام' if image_type == 'bone' else 'أسنان'}]"
        if user_message:
            user_msg += f"\n[رسالة مرفقة]: {user_message}"
        save_chat_message(patient_id, "user", user_msg)
        save_chat_message(patient_id, "assistant", report)
        trim_chat_messages(patient_id, MAX_HISTORY_MESSAGES)
        
        # Detect severity from analysis result
        severity = "low"
        urgency = str(analysis_result.get("urgency", "")).lower()
        if urgency in ["high", "critical", "مرتفع", "حرج"]:
            severity = "high"
        elif urgency in ["moderate", "medium", "متوسط"]:
            severity = "medium"
        else:
            ai_findings = analysis_result.get("ai_findings")
            if isinstance(ai_findings, list):
                severities = [str(item.get("severity", "")).upper() for item in ai_findings if isinstance(item, dict)]
                if any(s in ["HIGH", "CRITICAL"] for s in severities):
                    severity = "high"
                elif any(s in ["MODERATE", "MEDIUM"] for s in severities):
                    severity = "medium"
            elif isinstance(ai_findings, dict):
                sev = str(ai_findings.get("severity", "")).upper()
                if sev in ["HIGH", "CRITICAL"]:
                    severity = "high"
                elif sev in ["MODERATE", "MEDIUM"]:
                    severity = "medium"
            else:
                confidence_raw = analysis_result.get('confidence', 0)
                predicted_class = str(analysis_result.get('predicted_class', '')).lower()
                try:
                    if isinstance(confidence_raw, str):
                        confidence = float(confidence_raw.replace('%', '').strip())
                    else:
                        confidence = float(confidence_raw)
                except (ValueError, TypeError):
                    confidence = 0.0

                high_severity_conditions = ['caries', 'تسوس', 'fracture', 'كسر', 'gingivitis', 'التهاب']
                is_severe_condition = any(cond in predicted_class for cond in high_severity_conditions)

                if confidence > 80 and is_severe_condition:
                    severity = "high"
                elif confidence > 50:
                    severity = "medium"
        
        # Add follow-up suggestion based on severity
        # Note: booking question is handled by the frontend via suggest_booking flag
        # Only add care tips for medium severity (not booking prompt to avoid duplicates)
        if severity == "medium":
            if response_language == 'en':
                follow_up = "💡 Care tips: Maintain good oral hygiene, use mouthwash, and follow up with a doctor soon."
            else:
                follow_up = "💡 نصائح للعناية: حافظ على نظافة الأسنان واستخدم غسول الفم. تابع مع طبيب في أقرب وقت مناسب."
            report += f"\n\n{follow_up}"
        
        # Allow GradCAM visualization for chest X-ray only.
        gradcam_image = analysis_result.get('gradcam_image_base64', None) if image_type == 'chest' else None
        
        return jsonify({
            "success": True,
            "file_type": "image",
            "image_type": image_type,
            "analysis": analysis_result,
            "report": report,
            "patient_id": patient_id,
            "severity": severity,
            "suggest_booking": severity == "high",
            "gradcam_image": gradcam_image,
            "result_image": result_image
        })
    
    elif allowed_file(filename, ALLOWED_PDF_EXTENSIONS):
        # Extract text from PDF
        extraction_result = extract_pdf_text(file_bytes)
        
        if 'error' in extraction_result:
            return jsonify(extraction_result), 400
        
        # Generate summary report
        report = resolve_llm_response(
            generate_medical_report(extraction_result, 'pdf', target_language=response_language)
        )
        report = clean_markdown(report)

        if user_message:
            followup = generate_followup_for_uploaded_file(
                user_message=user_message,
                base_report=report,
                image_type=None,
                target_language=response_language,
            )
            if followup:
                followup_heading = "💬 Reply to your message:" if response_language == 'en' else "💬 رد على رسالتك:"
                report = f"{report}\n\n{followup_heading}\n{followup}"
        
        # Store in persistent conversation history
        user_msg = f"[تم رفع ملف PDF: {extraction_result.get('pages', 0)} صفحات]"
        if user_message:
            user_msg += f"\n[رسالة مرفقة]: {user_message}"
        save_chat_message(patient_id, "user", user_msg)
        save_chat_message(patient_id, "assistant", report)
        trim_chat_messages(patient_id, MAX_HISTORY_MESSAGES)
        
        return jsonify({
            "success": True,
            "file_type": "pdf",
            "pages": extraction_result.get('pages', 0),
            "text_preview": extraction_result.get('text', '')[:500] + "...",
            "report": report,
            "patient_id": patient_id
        })
    
    else:
        return jsonify({
            "error": f"نوع ملف غير مدعوم. الأنواع المدعومة: {', '.join(ALLOWED_IMAGE_EXTENSIONS | ALLOWED_PDF_EXTENSIONS)}"
        }), 400


@app.route('/api/prescription/status', methods=['GET'])
def prescription_status():
    """Proxy to the prescription parser /status so the UI can poll progress."""
    try:
        r = requests.get(f"{PRESCRIPTION_API}/status", timeout=5)
        if r.status_code == 200:
            return jsonify(r.json())
        return jsonify({"stage": "unknown", "message": f"HTTP {r.status_code}"}), 200
    except Exception as e:
        return jsonify({"stage": "unreachable", "message": str(e)}), 200


@app.route('/api/capabilities', methods=['GET'])
def get_capabilities():
    """Get chatbot capabilities and available features."""
    return jsonify({
        "image_support": IMAGE_SUPPORT,
        "pdf_support": PDF_SUPPORT,
        "llm_available": LLM_AVAILABLE,
        "bone_detect_api": BONE_DETECT_API,
        "oral_detect_api": ORAL_DETECT_API,
        "oral_classify_api": ORAL_CLASSIFY_API,
        "chest_xray_api": CHEST_XRAY_API,
        "allowed_image_types": list(ALLOWED_IMAGE_EXTENSIONS),
        "allowed_pdf_types": list(ALLOWED_PDF_EXTENSIONS),
        "max_file_size_mb": 64
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
