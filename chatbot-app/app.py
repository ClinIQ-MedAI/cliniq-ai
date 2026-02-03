#!/usr/bin/env python3
"""
Healthcare Chatbot Flask Application
Handles appointment booking, doctor availability, and general health questions.
"""

from flask import Flask, render_template, request, jsonify
import json
import os
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://llm.jetstream-cloud.org/api/")
MODEL = os.getenv("MODEL", "gpt-oss-120b")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

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
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_BASE_URL}chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            # Check if response has content before returning
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Validate that response actually has content
                    if data.get("choices") and len(data["choices"]) > 0:
                        message = data["choices"][0].get("message", {})
                        if message.get("content"):
                            return response  # Valid response with content
                except:
                    pass
                
                # If we got here, response was 200 but had no valid content
                if attempt < max_retries - 1:
                    wait_time = initial_wait * (2 ** attempt)
                    print(f"⚠️ Empty response (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            return response
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                print(f"⚠️ Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    return None


def get_llm_response(prompt: str, system_message: str = None, history: list = None):
    """Get response from LLM API using direct HTTP requests.
    
    Args:
        prompt: The user's message
        system_message: Optional system message for context
        history: Optional list of previous messages for conversation memory
    """
    if not LLM_AVAILABLE:
        # Return a default response for demonstration
        if "symptom" in prompt.lower() or "disease" in prompt.lower():
            return "I'm temporarily unable to access the LLM service. For health concerns, please consult with a doctor. Would you like to book an appointment instead?"
        return "I'm unable to access the LLM service at the moment. Please try booking an appointment with our doctors."
    
    messages = []
    
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Add conversation history for context
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": prompt})
    
    # Retry up to 4 times on empty response
    max_retries = 4
    
    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500,
            }
            
            response = make_api_request_with_retry(payload, headers)
            
            if response.status_code == 200:
                data = response.json()
                # Validate response structure
                if not data.get("choices") or len(data["choices"]) == 0:
                    print(f"DEBUG: Empty choices array (attempt {attempt + 1}/{max_retries}). Response: {data}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    return "Error: API returned no response. Please try again."
                
                message = data["choices"][0].get("message", {})
                content = message.get("content", "").strip()
                
                if not content:
                    print(f"DEBUG: Empty content (attempt {attempt + 1}/{max_retries}). Message: {message}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    return "Error: API returned empty response. Please try again later."
                
                # Success - return the response
                if attempt > 0:
                    print(f"✓ Got valid response on attempt {attempt + 1}")
                return clean_markdown(content)
            else:
                error_msg = f"API Error (status {response.status_code})"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f": {error_data['error']}"
                except:
                    error_msg += f": {response.text[:200]}"
                return error_msg
                
        except requests.exceptions.Timeout:
            return "The LLM service is taking too long to respond. Please try again later."
        except requests.exceptions.ConnectionError:
            return "Cannot connect to the LLM service. Please check your internet connection and try again."
        except Exception as e:
            return f"Error communicating with AI service: {str(e)}"
    
    return "Error: API returned empty response after multiple attempts. Please try again later."


def classify_query(user_message: str):
    """Classify the user's query type."""
    message_lower = user_message.lower()
    
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
- Then ask if they want more details.
  - In English: "Would you like more details about this?"
  - In Arabic: "هل تود معرفة المزيد من التفاصيل حول هذا الموضوع؟"
- If the user says yes, wants more info, or asks for details, THEN provide a comprehensive answer.
- If they ask a new question, give a brief answer to that instead.

IMPORTANT:
- You are NOT a doctor and cannot provide medical diagnosis or treatment.
- Always recommend consulting with a healthcare professional for specific medical concerns.
- Use numbered lists (1., 2., 3., etc) for symptoms/steps in detailed answers.
- Put each numbered item on a separate line.
- Avoid markdown formatting like **, -, or | characters. Use plain text only.
- You can reference previous messages in our conversation."""
    
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
    """Handle chat messages with conversation memory per patient."""
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
    
    # Route to appropriate handler with history
    if query_type == "health":
        response = handle_health_question(user_message, history)
    elif query_type == "appointment":
        response = handle_appointment_request(user_message, history)
    elif query_type == "availability":
        response = handle_availability_query(user_message, history)
    elif query_type == "faq":
        response = handle_faq(user_message, history)
    else:
        response = "I'm not sure how to help with that. Try asking about symptoms, booking an appointment, or general clinic information."
    
    # Store the exchange in history
    conversation_history[patient_id].append({"role": "user", "content": user_message})
    conversation_history[patient_id].append({"role": "assistant", "content": response})
    
    # Limit history size to prevent memory issues (keep last 50 messages)
    if len(conversation_history[patient_id]) > 50:
        conversation_history[patient_id] = conversation_history[patient_id][-50:]
    
    return jsonify({
        "response": response,
        "query_type": query_type,
        "patient_id": patient_id
    })


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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
