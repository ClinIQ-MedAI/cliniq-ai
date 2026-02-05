let doctors = [];
let selectedDoctor = null;
let isWaitingForResponse = false;
let selectedFile = null;

// Get or generate patient_id for this session
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
        
        const doctorSelect = document.getElementById('doctorSelect');
        doctorSelect.innerHTML = '<option value="">-- اختر طبيب --</option>';
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

function setLoadingState(isLoading, message = null) {
    isWaitingForResponse = isLoading;
    const input = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const fileInput = document.getElementById('fileInput');
    
    if (isLoading) {
        input.disabled = true;
        sendButton.disabled = true;
        fileInput.disabled = true;
        sendButton.textContent = '⏳';
        sendButton.style.opacity = '0.6';
        input.placeholder = message || 'جاري التحليل... / Analyzing...';
    } else {
        input.disabled = false;
        sendButton.disabled = false;
        fileInput.disabled = false;
        sendButton.textContent = '➤';
        sendButton.style.opacity = '1';
        input.placeholder = 'اكتب رسالتك هنا... / Type your message...';
        input.focus();
    }
}

// ==================== FILE UPLOAD FUNCTIONS ====================

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    selectedFile = file;
    const preview = document.getElementById('uploadPreview');
    const imagePreview = document.getElementById('imagePreview');
    const pdfPreview = document.getElementById('pdfPreview');
    const fileName = document.getElementById('fileName');
    const imageTypeSelector = document.getElementById('imageTypeSelector');
    
    fileName.textContent = file.name;
    preview.classList.remove('hidden');
    
    if (file.type.startsWith('image/')) {
        // Show image preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
        pdfPreview.classList.add('hidden');
        imageTypeSelector.classList.remove('hidden');
    } else if (file.type === 'application/pdf') {
        // Show PDF icon
        imagePreview.classList.add('hidden');
        pdfPreview.classList.remove('hidden');
        pdfPreview.textContent = `📄 ${file.name}`;
        imageTypeSelector.classList.add('hidden');
    }
}

function clearUpload() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('uploadPreview').classList.add('hidden');
    document.getElementById('imagePreview').classList.add('hidden');
    document.getElementById('pdfPreview').classList.add('hidden');
}

async function uploadFile() {
    if (!selectedFile) return null;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('patient_id', patientId);
    
    // Get selected image type (dental or bone)
    const imageTypeInput = document.querySelector('input[name="imageType"]:checked');
    const imageType = imageTypeInput ? imageTypeInput.value : 'dental';
    formData.append('image_type', imageType);
    
    // Show uploading message
    const fileType = selectedFile.type.startsWith('image/') ? 
        (imageType === 'bone' ? '🦴 صورة أشعة عظام' : '🦷 صورة أشعة أسنان') : 
        '📄 ملف PDF';
    addMessage(`[تم رفع ${fileType}]`, 'user');
    
    setLoadingState(true, 'جاري تحليل الملف...');
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            addMessage(data.report, 'bot');
        } else {
            addMessage(`خطأ: ${data.error}`, 'bot');
        }
        
        clearUpload();
        return data;
    } catch (error) {
        console.error('Upload error:', error);
        addMessage('حدث خطأ أثناء رفع الملف. حاول مرة أخرى.', 'bot');
        return null;
    } finally {
        setLoadingState(false);
    }
}

// ==================== MESSAGE FUNCTIONS ====================

function sendMessage(event) {
    event.preventDefault();
    
    if (isWaitingForResponse) return;
    
    // Check if there's a file to upload
    if (selectedFile) {
        uploadFile();
        return;
    }
    
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    addMessage(message, 'user');
    input.value = '';
    
    getResponse(message);
}

function addMessage(text, sender) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const p = document.createElement('p');
    // Preserve line breaks
    p.innerHTML = text.replace(/\n/g, '<br>');
    messageDiv.appendChild(p);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function getResponse(userMessage) {
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
        addMessage(data.response, 'bot');
        
        if (data.query_type === 'appointment') {
            setTimeout(() => {
                openBookingModal();
            }, 500);
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('عذراً، حدث خطأ. حاول مرة أخرى.', 'bot');
    } finally {
        setLoadingState(false);
    }
}

function quickAction(message) {
    if (isWaitingForResponse) return;
    const input = document.getElementById('messageInput');
    input.value = message;
    sendMessage(new Event('submit'));
}

// ==================== BOOKING FUNCTIONS ====================

function openBookingModal() {
    document.getElementById('bookingModal').classList.remove('hidden');
}

function closeBookingModal() {
    document.getElementById('bookingModal').classList.add('hidden');
}

function updateAvailability() {
    const doctorId = document.getElementById('doctorSelect').value;
    const daySelect = document.getElementById('daySelect');
    const timeSelect = document.getElementById('timeSelect');
    
    timeSelect.innerHTML = '<option value="">-- اختر وقت --</option>';
    daySelect.innerHTML = '<option value="">-- اختر يوم --</option>';
    
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
    
    timeSelect.innerHTML = '<option value="">-- اختر وقت --</option>';
    
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
        queueText.textContent = `ترتيبك في القائمة: ${data.queue_position} (${data.people_before_you} أشخاص قبلك)`;
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
        alert('يرجى ملء جميع الحقول');
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
            const confirmationMsg = `✅ تم حجز موعدك بنجاح!\n\nالتفاصيل:\nالطبيب: ${doctor.name}\nاليوم: ${date}\nالوقت: ${time}\nالاسم: ${patientName}\n\nنتطلع لرؤيتك!`;
            
            addMessage(confirmationMsg, 'bot');
            closeBookingModal();
            
            document.getElementById('bookingForm').reset();
            document.getElementById('queueInfo').classList.add('hidden');
        } else {
            alert('حدث خطأ في حجز الموعد. حاول مرة أخرى.');
        }
    } catch (error) {
        console.error('Error booking appointment:', error);
        alert('حدث خطأ في حجز الموعد. حاول مرة أخرى.');
    }
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('bookingModal');
    if (event.target == modal) {
        closeBookingModal();
    }
}
