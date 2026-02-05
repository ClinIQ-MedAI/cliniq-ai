#!/usr/bin/env python3
"""
Healthcare Chatbot Flask Application
Handles appointment booking, doctor availability, general health questions,
and medical image/PDF analysis.
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import json
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
    from PIL import Image
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
ORAL_CLASSIFY_API = "http://127.0.0.1:8002"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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
        elif image_type in ['dental', 'dental_photo', 'dental_xray']:
            api_url = f"{ORAL_CLASSIFY_API}/predict_for_llm"
        else:
            return {"error": f"Unknown image type: {image_type}"}

        
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(api_url, files=files, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to {image_type} API. Make sure it's running."}
    except Exception as e:
        return {"error": str(e)}


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


def generate_medical_report(analysis_result, file_type, image_type=None):
    """Generate medical report using LLM based on analysis results."""
    if not LLM_AVAILABLE:
        return f"نتائج التحليل:\n{json.dumps(analysis_result, ensure_ascii=False, indent=2)}"
    
    if file_type == 'image':
        if 'error' in analysis_result:
            return f"خطأ في التحليل: {analysis_result['error']}"
        
        prompt = f"""أنت طبيب متخصص. اكتب تقرير طبي مختصر بالعربية بناءً على نتائج التحليل.

نتائج التحليل:
{json.dumps(analysis_result, ensure_ascii=False, indent=2)}

اكتب التقرير بالشكل التالي (استخدم الرموز):

📋 التشخيص الرئيسي:
[اسم الحالة] - نسبة الثقة: [X]%

⚠️ الشدة: [منخفضة/متوسطة/عالية]

💊 التوصيات:
1. [توصية 1]
2. [توصية 2]
3. [توصية 3]

⏰ المتابعة: [متى يجب المتابعة]

اجعل التقرير مختصر وواضح. لا تستخدم جداول."""

    
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
        
        # Generate report
        report = generate_medical_report(analysis_result, 'image', image_type)
        
        # Store in conversation history
        if patient_id not in conversation_history:
            conversation_history[patient_id] = []
        
        user_msg = f"[تم رفع صورة أشعة {'عظام' if image_type == 'bone' else 'أسنان'}]"
        conversation_history[patient_id].append({"role": "user", "content": user_msg})
        conversation_history[patient_id].append({"role": "assistant", "content": report})
        
        # Detect severity from analysis result
        severity = "low"
        confidence_raw = analysis_result.get('confidence', 0)
        predicted_class = str(analysis_result.get('predicted_class', '')).lower()
        
        # Parse confidence to float, handling strings like "99.7%"
        try:
            if isinstance(confidence_raw, str):
                confidence = float(confidence_raw.replace('%', '').strip())
            else:
                confidence = float(confidence_raw)
        except (ValueError, TypeError):
            confidence = 0.0
            
        print(f"DEBUG: confidence_raw={confidence_raw}, parsed={confidence}, predicted_class={predicted_class}")
        
        # High severity conditions (confidence is percentage like 99.7, not decimal)
        high_severity_conditions = ['caries', 'تسوس', 'fracture', 'كسر', 'gingivitis', 'التهاب']
        is_severe_condition = any(cond in predicted_class for cond in high_severity_conditions)
        
        if confidence > 80 and is_severe_condition:
            severity = "high"
        elif confidence > 50:
            severity = "medium"
        
        print(f"DEBUG: severity={severity}, is_severe_condition={is_severe_condition}")
        
        # Add follow-up suggestion based on severity
        follow_up = None
        if severity == "high":
            follow_up = "⚠️ الحالة تحتاج متابعة سريعة. هل تريد حجز موعد مع طبيب الآن؟"
            report += f"\n\n{follow_up}"
        elif severity == "medium":
            follow_up = "💡 نصائح للعناية: حافظ على نظافة الأسنان واستخدم غسول الفم. تابع مع طبيب في أقرب وقت مناسب."
            report += f"\n\n{follow_up}"
        
        return jsonify({
            "success": True,
            "file_type": "image",
            "image_type": image_type,
            "analysis": analysis_result,
            "report": report,
            "patient_id": patient_id,
            "severity": severity,
            "suggest_booking": severity == "high"
        })
    
    elif allowed_file(filename, ALLOWED_PDF_EXTENSIONS):
        # Extract text from PDF
        extraction_result = extract_pdf_text(file_bytes)
        
        if 'error' in extraction_result:
            return jsonify(extraction_result), 400
        
        # Generate summary report
        report = generate_medical_report(extraction_result, 'pdf')
        
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
        "oral_classify_api": ORAL_CLASSIFY_API,
        "allowed_image_types": list(ALLOWED_IMAGE_EXTENSIONS),
        "allowed_pdf_types": list(ALLOWED_PDF_EXTENSIONS),
        "max_file_size_mb": 16
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
