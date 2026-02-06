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
from datetime import datetime
from pathlib import Path
import requests
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
    from PIL import Image, ImageDraw, ImageFont
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
DENTAL_XRAY_API = "http://127.0.0.1:8000"    # dental x-ray (YOLO + ConvNeXt pipeline)
ORAL_DETECT_API = "http://127.0.0.1:8002"    # oral disease detection (dental photos)
CHEST_XRAY_API = "http://127.0.0.1:8003"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_PDF_EXTENSIONS = {'pdf'}

# Conversation history storage per patient_id
# Format: {patient_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
conversation_history = {}

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


def handle_health_question(user_message: str, history: list = None):
    """Handle general health questions using LLM."""
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
    
    response = get_llm_response(user_message, system_message, history)
    return response


def handle_appointment_request(user_message: str, history: list = None):
    """Handle appointment booking requests."""
    doctors = load_json(DOCTORS_FILE)
    
    doctor_list = "\n".join([f"- {doc['name']} ({doc['specialty']})" for doc in doctors])
    
    prompt = f"""The user wants to book an appointment. Respond in the SAME LANGUAGE as the user (Arabic or English). Here are available doctors:
{doctor_list}

User message: {user_message}

Please ask which doctor they prefer and what day they'd like to schedule."""
    
    response = get_llm_response(prompt, None, history)
    return response


def handle_availability_query(user_message: str, history: list = None):
    """Handle doctor availability queries."""
    doctors = load_json(DOCTORS_FILE)
    
    doctor_info = json.dumps(doctors, indent=2)
    
    system_message = f"""You are a healthcare scheduling assistant. Respond in the SAME LANGUAGE as the user (Arabic or English). Here is the doctor availability information:
{doctor_info}

Answer the user's questions about doctor availability, specialties, and scheduling."""
    
    response = get_llm_response(user_message, system_message, history)
    return response


def handle_faq(user_message: str, history: list = None):
    """Handle FAQ queries."""
    faqs = load_json(FAQ_FILE)
    
    faq_text = "\n\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faqs])
    
    system_message = f"""You are a healthcare clinic assistant. Respond in the SAME LANGUAGE as the user (Arabic or English). Here are frequently asked questions and answers:

{faq_text}

Answer the user's question based on the FAQ information provided."""
    
    response = get_llm_response(user_message, system_message, history)
    return response


@app.route('/')
def index():
    """Serve the main chatbot page."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with conversation memory per patient (Streaming)."""
    data = request.json
    user_message = data.get('message', '').strip()
    patient_id = data.get('patient_id', 'anonymous')  # Default to 'anonymous' if not provided
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    # Initialize conversation history for this patient if not exists
    if patient_id not in conversation_history:
        conversation_history[patient_id] = []
    
    # Get patient's conversation history (limit to last 10 exchanges to avoid token limits)
    history = conversation_history[patient_id][-20:]  # Last 20 messages (10 exchanges)
    
    # Classify the query
    query_type = classify_query(user_message)
    
    def generate():
        # Route to appropriate handler with history
        if query_type == "upload":
            # For upload prompt, we just return the string immediately
            gen_response = "ممتاز! 📷 اضغط على زر 📎 بجانب مربع الكتابة لرفع الصورة أو ملف PDF.\n\nأنواع الملفات المدعومة:\n🦷 صور أشعة الأسنان\n🦴 صور أشعة العظام\n📄 تقارير طبية PDF"
        elif query_type == "health":
            gen_response = handle_health_question(user_message, history)
        elif query_type == "appointment":
            gen_response = handle_appointment_request(user_message, history)
        elif query_type == "availability":
            gen_response = handle_availability_query(user_message, history)
        elif query_type == "faq":
            gen_response = handle_faq(user_message, history)
        else:
            gen_response = "I'm not sure how to help with that. Try asking about symptoms, booking an appointment, or general clinic information."
        
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
        
        # Store the exchange in history
        conversation_history[patient_id].append({"role": "user", "content": user_message})
        conversation_history[patient_id].append({"role": "assistant", "content": full_response_text})
        
        # Limit history size
        if len(conversation_history[patient_id]) > 50:
            conversation_history[patient_id] = conversation_history[patient_id][-50:]
        
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
    
    if patient_id in conversation_history:
        conversation_history[patient_id] = []
    
    return jsonify({
        "success": True,
        "message": "Conversation history cleared",
        "patient_id": patient_id
    })


@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get conversation history for a patient."""
    patient_id = request.args.get('patient_id', 'anonymous')
    
    history = conversation_history.get(patient_id, [])
    
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
        elif image_type in ['dental', 'dental_photo']:
            api_url = f"{ORAL_DETECT_API}/predict_for_llm"
        elif image_type == 'dental_xray':
            # Dental X-Ray API on port 8000 uses /predict (no predict_for_llm)
            api_url = f"{DENTAL_XRAY_API}/predict"
        elif image_type == 'chest':
            api_url = f"{CHEST_XRAY_API}/predict_for_llm?include_gradcam=true"
        else:
            return {"error": f"Unknown image type: {image_type}"}

        
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(api_url, files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            # Transform dental xray raw /predict response into LLM-friendly format
            if image_type == 'dental_xray':
                result = transform_dental_xray_result(result)
            return result
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to {image_type} API. Make sure it's running."}
    except Exception as e:
        return {"error": str(e)}


# Clinical descriptions for dental xray classes
DENTAL_CLASS_INFO = {
    "Decay": {"ar": "تسوس الأسنان", "severity": "HIGH", "meaning": "تلف في بنية السن يتطلب علاج ترميمي (حشو أو تاج)"},
    "Apical Periodontitis": {"ar": "التهاب حوائط الذروة", "severity": "HIGH", "meaning": "التهاب حول جذر السن قد يستلزم علاج عصب"},
    "Wisdom Tooth": {"ar": "ضرس العقل", "severity": "MODERATE", "meaning": "قد يحتاج متابعة أو خلع إذا كان مطموراً أو يسبب مشاكل"},
    "Missing Tooth": {"ar": "سن مفقود", "severity": "MODERATE", "meaning": "قد يحتاج تعويض بزراعة أو جسر لمنع تحرك الأسنان المجاورة"},
    "Dental Filling": {"ar": "حشوة أسنان", "severity": "LOW", "meaning": "حشوة موجودة - يجب التأكد من سلامتها وعدم وجود تسوس ثانوي"},
    "Root Canal Filling": {"ar": "حشوة عصب", "severity": "LOW", "meaning": "علاج عصب سابق - يجب متابعة نجاح العلاج"},
    "Implant": {"ar": "زرعة أسنان", "severity": "LOW", "meaning": "زرعة موجودة - يجب التأكد من استقرارها وصحة العظم المحيط"},
    "Porcelain Crown": {"ar": "تاج خزفي", "severity": "LOW", "meaning": "تاج صناعي - يجب التأكد من ملاءمته وعدم وجود تسوس تحته"},
    "Ceramic Bridge": {"ar": "جسر خزفي", "severity": "LOW", "meaning": "جسر أسنان - يجب التأكد من سلامة الأسنان الداعمة"},
}


def transform_dental_xray_result(raw_result):
    """Transform raw dental xray /predict response into structured LLM-friendly format."""
    detections = raw_result.get("detections", [])
    
    if not detections:
        return {
            "patient_context": "تحليل صورة أشعة سينية للأسنان",
            "modality": "أشعة سينية بانورامية للأسنان",
            "body_part": "الفك والأسنان",
            "ai_findings": [],
            "urgency": "LOW",
            "summary": "لم يتم اكتشاف أي حالات في الصورة.",
            "recommendations": ["الصورة تبدو طبيعية. يُنصح بالمتابعة الدورية مع طبيب الأسنان."],
        }
    
    # Group detections by class and pick the highest confidence for each
    class_groups = {}
    for det in detections:
        # Use refined class if available, otherwise original
        cls = det.get("refined_class_name") or det.get("class_name", "Unknown")
        conf = det.get("refined_confidence") or det.get("confidence", 0)
        if cls not in class_groups or conf > class_groups[cls]["confidence"]:
            class_groups[cls] = {"confidence": conf, "count": class_groups.get(cls, {}).get("count", 0) + 1}
        else:
            class_groups[cls]["count"] = class_groups[cls].get("count", 0) + 1
    
    # Build structured findings
    ai_findings = []
    severities = []
    recommendations = []
    
    for cls, info in class_groups.items():
        cls_info = DENTAL_CLASS_INFO.get(cls, {"ar": cls, "severity": "MODERATE", "meaning": f"تم اكتشاف {cls}"})
        conf_pct = round(info["confidence"] * 100, 1)
        severity = cls_info["severity"]
        severities.append(severity)
        
        finding = {
            "finding": cls,
            "finding_ar": cls_info["ar"],
            "count": info["count"],
            "confidence": f"{conf_pct}%",
            "severity": severity,
            "clinical_meaning": cls_info["meaning"],
        }
        ai_findings.append(finding)
        recommendations.append(f"{cls_info['ar']} ({cls}): {cls_info['meaning']}")
    
    # Determine overall urgency
    if "HIGH" in severities:
        urgency = "HIGH"
    elif "MODERATE" in severities:
        urgency = "MODERATE"
    else:
        urgency = "LOW"
    
    # Build summary
    findings_text = ", ".join([
        f"{info['count']}x {DENTAL_CLASS_INFO.get(cls, {'ar': cls})['ar']}"
        for cls, info in class_groups.items()
    ])
    summary = f"تم اكتشاف {len(detections)} حالة في الصورة: {findings_text}."
    
    return {
        "patient_context": "تحليل صورة أشعة سينية للأسنان",
        "modality": "أشعة سينية بانورامية للأسنان (Dental X-Ray)",
        "body_part": "الفك والأسنان",
        "num_detections": raw_result.get("num_detections", len(detections)),
        "num_refined": raw_result.get("num_refined", 0),
        "ai_findings": ai_findings,
        "urgency": urgency,
        "summary": summary,
        "recommendations": recommendations,
    }


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


def draw_detections_on_image(image_bytes, detections):
    """Draw bbox detections on the image and return base64 JPEG string."""
    if not IMAGE_SUPPORT or not detections:
        return None

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = None
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
            label = det.get("label", "Finding")
            conf = det.get("confidence")
            if conf:
                label = f"{label} ({conf})"
            if font:
                # Use textbbox (Pillow 8.0+) instead of deprecated textsize
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                    text_w = right - left
                    text_h = bottom - top
                except AttributeError:
                    text_w, text_h = len(label) * 6, 12
                draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 6, y1], fill=(255, 80, 80))
                draw.text((x1 + 3, y1 - text_h - 4), label, fill=(255, 255, 255), font=font)

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG", quality=90)
        img_bytes.seek(0)
        return base64.b64encode(img_bytes.read()).decode()
    except Exception as e:
        print(f"⚠️ Error drawing detections: {e}")
        return None


def fetch_dental_xray_visualization(image_bytes):
    """Fetch annotated visualization from the Dental X-Ray API's /predict/visualize."""
    try:
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(
            f"{DENTAL_XRAY_API}/predict/visualize",
            files=files, timeout=60
        )
        if response.status_code == 200 and response.headers.get('content-type', '').startswith('image/'):
            return base64.b64encode(response.content).decode()
    except Exception as e:
        print(f"⚠️ Error fetching dental xray visualization: {e}")
    return None


def fetch_detection_overlay_detections(image_bytes, image_type):
    """Fetch bbox detections from detection APIs for overlay if missing."""
    if image_type == 'bone':
        api_url = f"{BONE_DETECT_API}/predict"
    elif image_type in ['dental', 'dental_photo']:
        api_url = f"{ORAL_DETECT_API}/predict"
    elif image_type == 'dental_xray':
        api_url = f"{DENTAL_XRAY_API}/predict"
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


def filter_low_confidence_findings(analysis_result, image_type):
    """Filter out low-confidence findings before sending to LLM.
    
    For chest xray: threshold 60% (multi-label classifier, low scores are noise)
    For detection models: threshold 30% (YOLO already filters internally)
    Returns (filtered_result, all_low_confidence: bool)
    """
    import copy
    result = copy.deepcopy(analysis_result)
    
    if image_type == 'chest':
        threshold = 0.60
        # Filter detected_conditions
        conditions = result.get('detected_conditions', [])
        if isinstance(conditions, list):
            result['detected_conditions'] = [
                c for c in conditions
                if _get_prob(c) >= threshold
            ]
        # Filter ai_findings if it's a dict with confidence
        ai_findings = result.get('ai_findings', {})
        if isinstance(ai_findings, dict):
            conf_str = str(ai_findings.get('confidence', '0%'))
            conf_val = float(conf_str.replace('%', '').strip()) / 100 if '%' in conf_str else float(conf_str)
            if conf_val < threshold:
                result['ai_findings'] = {'primary_diagnosis': 'No Finding', 'confidence': '100%', 'severity': 'LOW', 'clinical_meaning': 'لا توجد نتائج مقلقة'}
        # Filter differential_diagnoses
        diffs = result.get('differential_diagnoses', [])
        if isinstance(diffs, list):
            result['differential_diagnoses'] = [
                d for d in diffs
                if _get_prob(d) >= threshold
            ]
        # Check if everything was filtered out
        has_significant = bool(result.get('detected_conditions', []))
        if not has_significant:
            # Check ai_findings
            af = result.get('ai_findings', {})
            if isinstance(af, dict) and af.get('primary_diagnosis') != 'No Finding':
                has_significant = True
        return result, not has_significant
    
    elif image_type == 'dental_xray':
        # Already transformed by transform_dental_xray_result
        findings = result.get('ai_findings', [])
        if isinstance(findings, list):
            # Keep all detections (YOLO already filtered at conf 0.25)
            pass
        return result, not bool(result.get('ai_findings', []))
    
    else:
        # For bone, oral-detect — trust the API's own filtering
        return result, False


def _get_prob(item):
    """Extract probability from a conditions dict."""
    if not isinstance(item, dict):
        return 0
    prob_str = str(item.get('probability', item.get('confidence', '0')))
    try:
        val = float(prob_str.replace('%', '').strip())
        return val / 100 if val > 1 else val
    except (ValueError, TypeError):
        return 0


def generate_medical_report(analysis_result, file_type, image_type=None):
    """Generate medical report using LLM based on analysis results."""
    if not LLM_AVAILABLE:
        return f"نتائج التحليل:\n{json.dumps(analysis_result, ensure_ascii=False, indent=2)}"
    
    if file_type == 'image':
        if 'error' in analysis_result:
            return f"خطأ في التحليل: {analysis_result['error']}"
        
        image_type_labels = {
            'bone': 'أشعة عظام',
            'chest': 'أشعة صدر',
            'dental': 'أشعة أسنان',
            'dental_photo': 'صورة أسنان',
            'dental_xray': 'أشعة أسنان',
        }
        image_label = image_type_labels.get(image_type, 'صورة طبية')
        
        # Filter low-confidence findings
        filtered_result, all_low = filter_low_confidence_findings(analysis_result, image_type)
        
        if all_low:
            # All findings are below threshold — reassuring message
            prompt = f"""أنت طبيب متخصص. تم تحليل صورة {image_label} بالذكاء الاصطناعي ولم يتم اكتشاف أي حالة مرضية واضحة بنسبة ثقة كافية.

اكتب رسالة مطمئنة قصيرة بالعربية للمريض تتضمن:
✅ الأشعة لا تظهر أي نتائج مقلقة.
💡 نصيحة عامة بسيطة متعلقة بـ{image_label}.
⏰ متابعة دورية طبيعية.

اجعل الرسالة مختصرة ومطمئنة. لا تذكر نسب ثقة منخفضة. لا تستخدم جداول."""
        else:
            prompt = f"""أنت طبيب متخصص. اكتب تقرير طبي مختصر بالعربية بناءً على نتائج تحليل {image_label}.

نتائج التحليل:
{json.dumps(filtered_result, ensure_ascii=False, indent=2)}

اكتب التقرير بالشكل التالي (استخدم الرموز):

📋 النتائج المكتشفة:
اذكر كل حالة/كلاس تم اكتشافه في الصورة مع نسبة الثقة. لا تتجاهل أي detection.
شكل كل نتيجة:
• [اسم الحالة بالعربي] ([الاسم بالإنجليزي]) - نسبة الثقة: [X]%

⚠️ الشدة العامة: [منخفضة/متوسطة/عالية] (بناءً على أخطر حالة مكتشفة)

💊 التوصيات:
اكتب توصية مختصرة لكل حالة مكتشفة (توصية واحدة لكل حالة).

⏰ المتابعة: [متى يجب المتابعة]

مهم: التوصيات يجب أن تكون متعلقة بـ{image_label} تحديداً.
مهم: اذكر كل الحالات/الكلاسات المكتشفة وليس فقط الأعلى ثقة.
اجعل التقرير مختصر وواضح. لا تستخدم جداول. لا تضف نصائح عناية إضافية بعد المتابعة."""

    
    elif file_type == 'pdf':
        if 'error' in analysis_result:
            return f"خطأ في قراءة الملف: {analysis_result['error']}"
        
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


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads (images and PDFs) for medical analysis."""
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم إرسال ملف"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "لم يتم اختيار ملف"}), 400
    
    patient_id = request.form.get('patient_id', 'anonymous')
    image_type = request.form.get('image_type', 'dental')  # 'dental' or 'bone'
    
    filename = file.filename.lower()
    file_bytes = file.read()
    
    # Determine file type and process
    if allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
        if not IMAGE_SUPPORT:
            return jsonify({"error": "دعم الصور غير متاح"}), 500
        
        # Analyze image with appropriate API
        analysis_result = analyze_image_with_api(file_bytes, image_type)

        # Create annotated result image
        result_image = None
        if image_type == 'dental_xray':
            # Use the dental xray API's built-in visualization
            result_image = fetch_dental_xray_visualization(file_bytes)
        else:
            # For other types, draw bbox overlays manually
            detections = parse_detections_for_overlay(analysis_result)
            if not detections:
                detections = fetch_detection_overlay_detections(file_bytes, image_type)
            result_image = draw_detections_on_image(file_bytes, detections)
        
        # Generate report
        report = resolve_llm_response(
            generate_medical_report(analysis_result, 'image', image_type)
        )
        
        # Store in conversation history
        if patient_id not in conversation_history:
            conversation_history[patient_id] = []
        
        user_msg = f"[تم رفع صورة أشعة {'عظام' if image_type == 'bone' else 'أسنان'}]"
        conversation_history[patient_id].append({"role": "user", "content": user_msg})
        conversation_history[patient_id].append({"role": "assistant", "content": report})
        
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
            care_tips = {
                'bone': "💡 نصائح للعناية: تجنب إجهاد المنطقة المصابة واستخدم الكمادات الباردة. تابع مع طبيب عظام في أقرب وقت مناسب.",
                'chest': "💡 نصائح للعناية: تجنب التدخين والتعرض للملوثات. حافظ على التهوية الجيدة وتابع مع طبيب أمراض صدرية في أقرب وقت.",
                'dental': "💡 نصائح للعناية: حافظ على نظافة الأسنان واستخدم غسول الفم. تابع مع طبيب أسنان في أقرب وقت مناسب.",
                'dental_photo': "💡 نصائح للعناية: حافظ على نظافة الأسنان واستخدم غسول الفم. تابع مع طبيب أسنان في أقرب وقت مناسب.",
                'dental_xray': "💡 نصائح للعناية: حافظ على نظافة الأسنان واستخدم غسول الفم. تابع مع طبيب أسنان في أقرب وقت مناسب.",
            }
            follow_up = care_tips.get(image_type, "💡 تابع مع طبيب مختص في أقرب وقت مناسب.")
            report += f"\n\n{follow_up}"
        
        # Include GradCAM image in response if available (for chest x-ray)
        gradcam_image = analysis_result.get('gradcam_image_base64', None)
        
        return jsonify({
            "success": True,
            "file_type": "image",
            "image_type": image_type,
            "analysis": analysis_result,
            "report": report,
            "patient_id": patient_id,
            "severity": severity,
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
            generate_medical_report(extraction_result, 'pdf')
        )
        
        # Store in conversation history
        if patient_id not in conversation_history:
            conversation_history[patient_id] = []
        
        user_msg = f"[تم رفع ملف PDF: {extraction_result.get('pages', 0)} صفحات]"
        conversation_history[patient_id].append({"role": "user", "content": user_msg})
        conversation_history[patient_id].append({"role": "assistant", "content": report})
        
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


@app.route('/api/capabilities', methods=['GET'])
def get_capabilities():
    """Get chatbot capabilities and available features."""
    return jsonify({
        "image_support": IMAGE_SUPPORT,
        "pdf_support": PDF_SUPPORT,
        "llm_available": LLM_AVAILABLE,
        "bone_detect_api": BONE_DETECT_API,
        "dental_xray_api": DENTAL_XRAY_API,
        "oral_detect_api": ORAL_DETECT_API,
        "chest_xray_api": CHEST_XRAY_API,
        "allowed_image_types": list(ALLOWED_IMAGE_EXTENSIONS),
        "allowed_pdf_types": list(ALLOWED_PDF_EXTENSIONS),
        "max_file_size_mb": 64
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
