let doctors = [];
let selectedDoctor = null;
let isWaitingForResponse = false; // Track if waiting for bot response

// Get or generate patient_id for this session
// In production, this would come from the external auth backend
let patientId = localStorage.getItem('patient_id');
if (!patientId) {
    patientId = 'patient_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('patient_id', patientId);
}
console.log('Patient ID:', patientId);

// Load doctors on page load
document.addEventListener('DOMContentLoaded', function() {
    loadDoctors();
    
    // Scroll to bottom of messages
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
});

async function loadDoctors() {
    try {
        const response = await fetch('/api/doctors');
        doctors = await response.json();
        
        // Populate doctor select in modal
        const doctorSelect = document.getElementById('doctorSelect');
        doctorSelect.innerHTML = '<option value="">-- Choose a doctor --</option>';
        doctors.forEach(doctor => {
            const option = document.createElement('option');
            option.value = doctor.id;
            option.textContent = `${doctor.name} (${doctor.specialty})`;
            doctorSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading doctors:', error);
    }
}

function setLoadingState(isLoading) {
    isWaitingForResponse = isLoading;
    const input = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    if (isLoading) {
        input.disabled = true;
        sendButton.disabled = true;
        sendButton.textContent = '⏳';
        sendButton.style.opacity = '0.6';
        input.placeholder = 'انتظر الرد... / Waiting for response...';
    } else {
        input.disabled = false;
        sendButton.disabled = false;
        sendButton.textContent = '➤';
        sendButton.style.opacity = '1';
        input.placeholder = 'اكتب رسالتك هنا... / Type your message...';
        input.focus();
    }
}

function sendMessage(event) {
    event.preventDefault();
    
    // Prevent sending while waiting for response
    if (isWaitingForResponse) return;
    
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Display user message
    addMessage(message, 'user');
    input.value = '';
    
    // Get bot response
    getResponse(message);
}

function addMessage(text, sender) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const p = document.createElement('p');
    p.textContent = text;
    messageDiv.appendChild(p);
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function getResponse(userMessage) {
    // Set loading state
    setLoadingState(true);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: userMessage,
                patient_id: patientId
            })
        });
        
        const data = await response.json();
        const botResponse = data.response;
        const queryType = data.query_type;
        
        // Display bot response
        addMessage(botResponse, 'bot');
        
        // If booking was requested, show booking modal
        if (queryType === 'appointment') {
            setTimeout(() => {
                openBookingModal();
            }, 500);
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your request. Please try again.', 'bot');
    } finally {
        // Always reset loading state
        setLoadingState(false);
    }
}

function quickAction(message) {
    if (isWaitingForResponse) return; // Prevent quick actions while waiting
    const input = document.getElementById('messageInput');
    input.value = message;
    sendMessage(new Event('submit'));
}

function openBookingModal() {
    const modal = document.getElementById('bookingModal');
    modal.classList.remove('hidden');
}

function closeBookingModal() {
    const modal = document.getElementById('bookingModal');
    modal.classList.add('hidden');
}

function updateAvailability() {
    const doctorId = document.getElementById('doctorSelect').value;
    const daySelect = document.getElementById('daySelect');
    const timeSelect = document.getElementById('timeSelect');
    
    timeSelect.innerHTML = '<option value="">-- Choose a time --</option>';
    daySelect.innerHTML = '<option value="">-- Choose a day --</option>';
    
    if (!doctorId) return;
    
    selectedDoctor = doctors.find(d => d.id == doctorId);
    
    if (selectedDoctor) {
        const days = Object.keys(selectedDoctor.availability);
        days.forEach(day => {
            const option = document.createElement('option');
            option.value = day;
            option.textContent = day;
            daySelect.appendChild(option);
        });
    }
}

function updateTimeSlots() {
    const day = document.getElementById('daySelect').value;
    const timeSelect = document.getElementById('timeSelect');
    
    timeSelect.innerHTML = '<option value="">-- Choose a time --</option>';
    
    if (!day || !selectedDoctor) return;
    
    const times = selectedDoctor.availability[day] || [];
    times.forEach(time => {
        const option = document.createElement('option');
        option.value = time;
        option.textContent = time;
        timeSelect.appendChild(option);
    });
}

async function checkQueue() {
    const doctorId = document.getElementById('doctorSelect').value;
    const day = document.getElementById('daySelect').value;
    const time = document.getElementById('timeSelect').value;
    const queueInfo = document.getElementById('queueInfo');
    
    if (!doctorId || !day || !time) {
        queueInfo.classList.add('hidden');
        return;
    }
    
    try {
        const response = await fetch(
            `/api/appointments/queue?doctor_id=${doctorId}&date=${day}&time=${time}`
        );
        const data = await response.json();
        
        const queueText = document.getElementById('queueText');
        queueText.textContent = `Position in queue: ${data.queue_position} (${data.people_before_you} people before you)`;
        queueInfo.classList.remove('hidden');
    } catch (error) {
        console.error('Error checking queue:', error);
    }
}

async function submitBooking(event) {
    event.preventDefault();
    
    const patientName = document.getElementById('patientName').value;
    const doctorId = document.getElementById('doctorSelect').value;
    const date = document.getElementById('daySelect').value;
    const time = document.getElementById('timeSelect').value;
    
    if (!patientName || !doctorId || !date || !time) {
        alert('Please fill in all fields');
        return;
    }
    
    try {
        const response = await fetch('/api/appointments/book', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                patient_id: patientId,
                patient_name: patientName,
                doctor_id: parseInt(doctorId),
                date: date,
                time: time
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const doctor = doctors.find(d => d.id == doctorId);
            const confirmationMsg = `Great! Your appointment has been booked!\n\nDetails:\nDoctor: ${doctor.name}\nDate: ${date}\nTime: ${time}\nPatient: ${patientName}\n\nWe look forward to seeing you!`;
            
            addMessage(confirmationMsg, 'bot');
            closeBookingModal();
            
            // Reset form
            document.getElementById('bookingForm').reset();
            document.getElementById('queueInfo').classList.add('hidden');
        } else {
            alert('Error booking appointment. Please try again.');
        }
    } catch (error) {
        console.error('Error booking appointment:', error);
        alert('Error booking appointment. Please try again.');
    }
}

// Close modal when clicking outside of it
window.onclick = function(event) {
    const modal = document.getElementById('bookingModal');
    if (event.target == modal) {
        closeBookingModal();
    }
}
