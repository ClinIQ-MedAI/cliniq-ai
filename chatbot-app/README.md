# Healthcare Chatbot

A Flask-based conversational AI chatbot designed to assist healthcare clinic operations. The chatbot intelligently handles appointment bookings, doctor availability queries, FAQ responses, and general health information.

## Features

### 🤖 Intelligent Query Classification
- **Appointment Booking** - Recognizes appointment booking requests and guides users through the booking process
- **Doctor Availability** - Provides information about doctor schedules and specialties
- **FAQ Handling** - Answers frequently asked questions about the clinic
- **Health Questions** - Provides general health information and wellness guidance

### 📅 Appointment Management
- View available doctors and their specialties
- Book appointments with specific doctors
- Check queue positions and wait times
- Store appointments persistently in JSON

### 🏥 Doctor Management
- Maintain a database of doctors with specialties
- Display doctor availability information
- Manage doctor profiles and contact details

### 📚 Knowledge Base
- Frequently asked questions about clinic operations
- Insurance information
- Hours of operation
- Appointment rescheduling policies
- Telehealth options

## Project Structure

```
chatbot-app/
├── app.py                 # Flask application and API endpoints
├── requirements.txt       # Python dependencies
├── data/
│   ├── doctors.json      # Doctor profiles and availability
│   ├── appointments.json # Booked appointments
│   └── faq.json          # Frequently asked questions
├── static/
│   ├── script.js         # Frontend JavaScript
│   └── style.css         # Frontend styling
└── templates/
    └── index.html        # Chat interface
```

## Installation

### Prerequisites
- Python 3.8+
- Flask
- Requests

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API credentials:
   - Edit `app.py` and update `API_KEY` with your LLM API key
   - Update `API_BASE_URL` and `MODEL` as needed

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## API Endpoints

### Chat Endpoint
- **POST** `/api/chat`
  - Send a message and receive an intelligent response
  - Request: `{"message": "user message"}`
  - Response: `{"response": "chatbot response", "query_type": "health|appointment|availability|faq"}`

### Doctor Information
- **GET** `/api/doctors`
  - Retrieve list of all available doctors
  - Response: Array of doctor objects

### Appointment Booking
- **POST** `/api/appointments/book`
  - Book an appointment
  - Request: `{"patient_name": "John Doe", "doctor_id": 1, "date": "2026-02-15", "time": "10:00"}`
  - Response: Confirmation with appointment details

### Queue Information
- **GET** `/api/appointments/queue?doctor_id=1&date=2026-02-15&time=10:00`
  - Check queue position and wait time
  - Response: `{"queue_position": 3, "people_before_you": 2}`

## Data Files

### doctors.json
Contains doctor profiles with specialty and availability information.

Example:
```json
[
  {
    "id": 1,
    "name": "Dr. John Smith",
    "specialty": "General Practitioner",
    "availability": ["Monday-Friday 9AM-5PM"],
    "phone": "555-0100"
  }
]
```

### appointments.json
Stores all booked appointments with patient and scheduling details.

### faq.json
Contains frequently asked questions and answers about the clinic.

Example:
```json
[
  {
    "question": "What are your clinic hours?",
    "answer": "We are open Monday to Friday, 9 AM to 5 PM"
  }
]
```

## Configuration

Edit the following in `app.py` to customize:

```python
API_KEY = "your-api-key"              # LLM API authentication
API_BASE_URL = "https://api.url/"     # LLM API endpoint
MODEL = "model-name"                  # LLM model to use
```

## Usage Examples

### Booking an Appointment
```
User: "I want to book an appointment"
Chatbot: "Sure! Here are our available doctors: [list]. Which doctor would you prefer?"
```

### Doctor Availability
```
User: "When is Dr. Smith available?"
Chatbot: "Dr. Smith is available Monday to Friday from 9 AM to 5 PM"
```

### Health Questions
```
User: "What should I do about a fever?"
Chatbot: "A fever can be caused by various conditions. Please consult with a healthcare professional for proper diagnosis..."
```

### FAQ
```
User: "What is your clinic's phone number?"
Chatbot: "You can reach us at [clinic contact]..."
```

## Technologies Used

- **Framework**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Database**: JSON files
- **LLM**: Jetstream Cloud LLM API
- **API Calls**: Python Requests library

## Error Handling

The chatbot gracefully handles:
- API connectivity issues
- Invalid requests
- Missing data files
- LLM service unavailability

## Future Enhancements

- Database integration (PostgreSQL/MongoDB)
- Multi-language support
- SMS/WhatsApp integration
- Email notifications for appointments
- Doctor reviews and ratings
- Prescription management
- Medical records integration

## License

Proprietary - All rights reserved.

## Input Gate (OOD screening)

Before any model runs, every upload passes through a training-free screen
(`imaging/gate.py`) that rejects obviously-wrong inputs, so a colour selfie sent
as a bone X-ray is refused with a reason instead of being confidently scored.

It is deliberately conservative (only rejects when confident) and checks:

- **Grayscale-ness** — X-ray modalities (bone, chest, dental X-ray) are near
  grayscale; a saturated colour photo almost certainly is not an X-ray
  (`>15%` coloured pixels on an X-ray => reject).
- **Degeneracy** — blank / near-constant frames (intensity std `< 3`), extreme
  aspect ratios (`> 5:1`, banner/screenshot-like), or images too small to carry
  diagnostic information.

A rejected upload returns `200` with `input_rejected: true` and a human-readable
reason; there is no `report` field. The gate is wired in
`core/medical.py:_screen_upload()` and called at the top of `/api/upload`.

