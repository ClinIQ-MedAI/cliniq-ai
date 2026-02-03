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

# Configuration
API_KEY = "sk-0b8d90d6b3a44c64a5816df734975a62"
API_BASE_URL = "https://llm.jetstream-cloud.org/api/"
MODEL = "gpt-oss-120b"

# Initialize Flask app
app = Flask(__name__)

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


def get_llm_response(prompt: str, system_message: str = None):
    """Get response from LLM API using direct HTTP requests."""
    if not LLM_AVAILABLE:
        # Return a default response for demonstration
        if "symptom" in prompt.lower() or "disease" in prompt.lower():
            return "I'm temporarily unable to access the LLM service. For health concerns, please consult with a doctor. Would you like to book an appointment instead?"
        return "I'm unable to access the LLM service at the moment. Please try booking an appointment with our doctors."
    
    messages = []
    
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    messages.append({"role": "user", "content": prompt})
    
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
        
        response = requests.post(
            f"{API_BASE_URL}chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"Error from API (status {response.status_code}): {response.text}"
            
    except requests.exceptions.Timeout:
        return "The LLM service is taking too long to respond. Please try again."
    except Exception as e:
        return f"Error communicating with AI service: {str(e)}"


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


def handle_health_question(user_message: str):
    """Handle general health questions using LLM."""
    system_message = """You are a helpful healthcare information assistant. You provide general health information and answer questions about diseases, symptoms, and wellness. 
    IMPORTANT: You are NOT a doctor and cannot provide medical diagnosis or treatment. Always recommend consulting with a healthcare professional for specific medical concerns.
    Keep responses concise and informative."""
    
    response = get_llm_response(user_message, system_message)
    return response


def handle_appointment_request(user_message: str):
    """Handle appointment booking requests."""
    doctors = load_json(DOCTORS_FILE)
    
    doctor_list = "\n".join([f"- {doc['name']} ({doc['specialty']})" for doc in doctors])
    
    prompt = f"""The user wants to book an appointment. Here are available doctors:
{doctor_list}

User message: {user_message}

Please ask which doctor they prefer and what day they'd like to schedule."""
    
    response = get_llm_response(prompt)
    return response


def handle_availability_query(user_message: str):
    """Handle doctor availability queries."""
    doctors = load_json(DOCTORS_FILE)
    
    doctor_info = json.dumps(doctors, indent=2)
    
    system_message = f"""You are a healthcare scheduling assistant. Here is the doctor availability information:
{doctor_info}

Answer the user's questions about doctor availability, specialties, and scheduling."""
    
    response = get_llm_response(user_message, system_message)
    return response


def handle_faq(user_message: str):
    """Handle FAQ queries."""
    faqs = load_json(FAQ_FILE)
    
    faq_text = "\n\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faqs])
    
    system_message = f"""You are a healthcare clinic assistant. Here are frequently asked questions and answers:

{faq_text}

Answer the user's question based on the FAQ information provided."""
    
    response = get_llm_response(user_message, system_message)
    return response


@app.route('/')
def index():
    """Serve the main chatbot page."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    # Classify the query
    query_type = classify_query(user_message)
    
    # Route to appropriate handler
    if query_type == "health":
        response = handle_health_question(user_message)
    elif query_type == "appointment":
        response = handle_appointment_request(user_message)
    elif query_type == "availability":
        response = handle_availability_query(user_message)
    elif query_type == "faq":
        response = handle_faq(user_message)
    else:
        response = "I'm not sure how to help with that. Try asking about symptoms, booking an appointment, or general clinic information."
    
    return jsonify({
        "response": response,
        "query_type": query_type
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
