# Healthcare Chatbot

A web-based healthcare chatbot that helps users with:
- General health information questions
- Appointment booking with doctors
- Doctor availability checking
- Clinic FAQ and general information

## Features

✅ **Health Information**: Ask about diseases, symptoms, and wellness  
✅ **Appointment Booking**: Book appointments with available doctors  
✅ **Doctor Availability**: Check which doctors are available and when  
✅ **Queue Status**: See how many people are ahead of you  
✅ **Clinic FAQ**: Get answers about clinic policies and procedures  

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the application**:
```bash
python app.py
```

3. **Open in browser**:
Navigate to `http://localhost:5000`

## Project Structure

```
healthcare-chatbot/
├── app.py                 # Flask application & backend logic
├── requirements.txt       # Python dependencies
├── data/
│   ├── doctors.json      # Doctor information & availability
│   ├── appointments.json # Booked appointments
│   └── faq.json          # Frequently asked questions
├── templates/
│   └── index.html        # Chat interface
└── static/
    ├── style.css         # Chatbot styling
    └── script.js         # Frontend logic
```

## How It Works

1. **Query Classification**: The chatbot analyzes user input to determine if it's:
   - Health question → Uses LLM for general health info
   - Appointment booking → Opens booking interface
   - Doctor availability → Shows available slots
   - FAQ → Provides clinic information

2. **Appointment Booking**: Users can:
   - Select a doctor
   - Choose a date
   - Pick a time slot
   - See queue position in real-time

3. **Data Persistence**: All appointments are saved to `data/appointments.json`

## Customization

### Add More Doctors
Edit `data/doctors.json` and add doctor entries with availability schedules.

### Add More FAQ
Edit `data/faq.json` to add new questions and answers.

### Change the Model
In `app.py`, modify the `MODEL` variable to use different LLM models (e.g., "DeepSeek-R1", "llama-4-scout").

## Features to Add Later

- Email/SMS notifications for appointments
- Patient history tracking
- Doctor reviews and ratings
- Prescription management
- Telemedicine integration
- Multiple language support

---

Built with Flask, OpenAI API, and modern web technologies.
