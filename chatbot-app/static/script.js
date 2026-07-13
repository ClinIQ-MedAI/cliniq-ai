let doctors = [];
let selectedDoctor = null;
let isWaitingForResponse = false;
let selectedFile = null;
let uploadPickerMode = 'manual';
const DEFAULT_FILE_ACCEPT = '.jpg,.jpeg,.png,.gif,.bmp,.webp,.pdf';
const UPLOAD_TYPE_ACCEPT = {
    dental_photo: 'image/*',
    dental_xray: 'image/*',
    bone: 'image/*',
    chest: 'image/*',
    prescription: '.jpg,.jpeg,.png,.gif,.bmp,.webp,.pdf',
    pdf_report: '.pdf'
};

const I18N = {
    ar: {
        locale: 'ar-EG',
        dir: 'rtl',
        pageTitle: 'مساعد الرعاية الصحية | Healthcare Assistant',
        appSubtitle: 'مساعدك الصحي الذكي',
        sidebarTitle: 'المحادثات',
        newChat: 'محادثة جديدة',
        noChats: 'لا توجد محادثات بعد',
        chatFallbackTitle: 'محادثة',
        toggleSidebarTitle: 'إخفاء/إظهار',
        openSidebarTitle: 'عرض المحادثات',
        languageSelectTitle: 'اللغة الافتراضية عند رفع ملف بدون رسالة',
        clearCurrentChatTitle: 'مسح هيستوري المحادثة الحالية',
        exportChatTitle: 'تصدير المحادثة',
        toggleThemeTitle: 'تبديل الوضع',
        welcomeMessage: 'مرحباً! 👋 كيف أقدر أساعدك؟',
        messagePlaceholder: 'اكتب رسالتك هنا... / Type your message...',
        analyzingPlaceholder: 'جاري التحليل... / Analyzing...',
        pinUploadTitle: 'ارفع صورة أو PDF',
        quickFluLabel: '🤒 الإنفلونزا',
        quickBookingLabel: '📅 حجز موعد',
        quickScanLabel: '🩻 رفع أشعة',
        quickManualUploadLabel: '📎 رفع يدوي',
        quickHoursLabel: '🕐 ساعات العمل',
        quickPromptFlu: 'أخبرني عن أعراض الإنفلونزا',
        quickPromptAppointment: 'أريد حجز موعد',
        quickPromptHours: 'ما هي ساعات عمل العيادة؟',
        uploadTypeLabels: {
            dental_xray: '🩻 أشعة أسنان',
            bone: '🦴 أشعة عظام',
            chest: '🫁 أشعة صدر',
            dental_photo: '🦷 صورة أسنان',
            prescription: '📝 روشتة طبية',
            pdf_report: '📄 تقرير PDF',
            manual: '📎 رفع يدوي'
        },
        uploadTypeTitle: 'اختر نوع الملف، وهتفتح نافذة اختيار الملف مباشرة:',
        uploadWaitCurrentToast: '⏳ انتظر انتهاء التحليل الحالي ثم ارفع ملفًا جديدًا',
        uploadPickNowToast: 'حدد الملف الآن وسيتم رفعه مباشرة',
        uploadErrorGeneric: 'حدث خطأ أثناء رفع الملف. حاول مرة أخرى.',
        uploadReadError: 'حدث خطأ أثناء الرفع أو الاتصال بالخدمة.',
        uploadReceivedTag: '[تم رفع {fileType}]',
        imagePreviewAlt: 'صورة مرفوعة',
        clickToZoomTitle: 'اضغط للتكبير',
        uploadProgressTitle: '⏳ جاري تحليل {fileType}',
        uploadProgressStart: 'تم استلام الملف. جاري البدء...',
        uploadProgressPreparing: 'تم استلام الملف. جاري التحضير...',
        uploadAnalyzingFile: 'جاري تحليل الملف...',
        uploadStageUpload: 'جاري رفع الملف للخادم...',
        uploadStagePreprocess: 'جاري المعالجة الأولية للصورة...',
        uploadStageModel: 'جاري تشغيل النموذج الطبي...',
        uploadStageFinal: 'جاري تجهيز التقرير النهائي...',
        uploadStageAnalyze: 'جاري تحليل الصورة...',
        uploadPreparingResult: 'جاري تجهيز النتيجة النهائية...',
        uploadDone: 'تم التحليل بنجاح.',
        uploadAnalyzeFailed: 'تعذر التحليل: {error}',
        unknownError: 'خطأ غير معروف',
        errorPrefix: 'خطأ: {error}',
        prescriptionElapsedSec: ' ({seconds}ث)',
        prescriptionDefaultStatus: 'جاري المعالجة...',
        prescriptionErrorStatus: 'حدث خطأ أثناء تحليل الروشتة{elapsed}',
        modelResultHeading: '🧠 ناتج تحليل النموذج:',
        modelResultAlt: 'نتيجة التحليل',
        gradcamHeading: '📊 خريطة التركيز (Grad-CAM):',
        gradcamAlt: 'تصور Grad-CAM',
        gradcamHint: 'المناطق الحمراء تشير إلى المناطق التي ركز عليها النموذج في التشخيص',
        chatErrorGeneric: 'عذراً، حدث خطأ. حاول مرة أخرى.',
        bookingTitle: 'حجز موعد',
        patientNameLabel: 'الاسم:',
        doctorLabel: 'اختر الطبيب:',
        dayLabel: 'اختر اليوم:',
        timeLabel: 'اختر الوقت:',
        selectDoctorPlaceholder: '-- اختر طبيب --',
        selectDayPlaceholder: '-- اختر يوم --',
        selectTimePlaceholder: '-- اختر وقت --',
        bookingSubmit: 'تأكيد الحجز',
        queueText: 'ترتيبك في القائمة: {queue} ({people} أشخاص قبلك)',
        fillAllFields: 'يرجى ملء جميع الحقول',
        bookingErrorAlert: 'حدث خطأ في حجز الموعد. حاول مرة أخرى.',
        bookingSuccessMessage: '✅ تم حجز موعدك بنجاح!\n\nالتفاصيل:\nالطبيب: {doctor}\nاليوم: {date}\nالوقت: {time}\nالاسم: {name}\n\nنتطلع لرؤيتك!',
        clearChatConfirm: 'هل تريد مسح كل رسائل هذه المحادثة؟',
        clearChatSuccessToast: 'تم مسح هيستوري المحادثة',
        clearChatFailToast: 'تعذر مسح الهيستوري',
        deleteChatConfirm: 'هل تريد حذف هذه المحادثة؟',
        deleteChatToast: 'تم حذف المحادثة',
        deleteLabel: 'حذف',
        exportYouLabel: 'أنت',
        exportDoneToast: 'تم تصدير المحادثة',
        copyButtonLabel: 'نسخ',
        copyDoneToast: 'تم النسخ ✓',
        copyFailToast: 'تعذر النسخ',
        imageTypeDentalPhotoLabel: '🦷 صورة أسنان',
        imageTypeDentalXrayLabel: '🩻 أشعة أسنان',
        imageTypeBoneLabel: '🦴 أشعة عظام',
        imageTypeChestLabel: '🫁 أشعة صدر',
        imageTypePrescriptionLabel: '📝 روشتة',
        dayNames: {
            monday: 'الاثنين',
            tuesday: 'الثلاثاء',
            wednesday: 'الأربعاء',
            thursday: 'الخميس',
            friday: 'الجمعة',
            saturday: 'السبت',
            sunday: 'الأحد'
        }
    },
    en: {
        locale: 'en-US',
        dir: 'ltr',
        pageTitle: 'Healthcare Assistant | مساعد الرعاية الصحية',
        appSubtitle: 'Your smart healthcare assistant',
        sidebarTitle: 'Conversations',
        newChat: 'New chat',
        noChats: 'No conversations yet',
        chatFallbackTitle: 'Chat',
        toggleSidebarTitle: 'Hide/Show sidebar',
        openSidebarTitle: 'Show conversations',
        languageSelectTitle: 'Default language when uploading without text',
        clearCurrentChatTitle: 'Clear current chat history',
        exportChatTitle: 'Export chat',
        toggleThemeTitle: 'Toggle theme',
        welcomeMessage: 'Hello! 👋 How can I help you?',
        messagePlaceholder: 'Type your message here...',
        analyzingPlaceholder: 'Analyzing...',
        pinUploadTitle: 'Upload image or PDF',
        quickFluLabel: '🤒 Flu',
        quickBookingLabel: '📅 Book appointment',
        quickScanLabel: '🩻 Scan upload',
        quickManualUploadLabel: '📎 Manual upload',
        quickHoursLabel: '🕐 Working hours',
        quickPromptFlu: 'Tell me about flu symptoms',
        quickPromptAppointment: 'I want to book an appointment',
        quickPromptHours: 'What are the clinic working hours?',
        uploadTypeLabels: {
            dental_xray: '🩻 Dental X-ray',
            bone: '🦴 Bone X-ray',
            chest: '🫁 Chest X-ray',
            dental_photo: '🦷 Dental photo',
            prescription: '📝 Prescription (Image/PDF)',
            pdf_report: '📄 Medical PDF report',
            manual: '📎 Manual upload'
        },
        uploadTypeTitle: 'Choose file type, and the file picker will open immediately:',
        uploadWaitCurrentToast: '⏳ Wait for the current analysis to finish before uploading a new file',
        uploadPickNowToast: 'Choose the file now and it will upload immediately',
        uploadErrorGeneric: 'An error occurred while uploading the file. Please try again.',
        uploadReadError: 'An upload or service connection error occurred.',
        uploadReceivedTag: '[Uploaded {fileType}]',
        imagePreviewAlt: 'Uploaded image',
        clickToZoomTitle: 'Click to zoom',
        uploadProgressTitle: '⏳ Analyzing {fileType}',
        uploadProgressStart: 'File received. Starting...',
        uploadProgressPreparing: 'File received. Preparing...',
        uploadAnalyzingFile: 'Analyzing file...',
        uploadStageUpload: 'Uploading file to server...',
        uploadStagePreprocess: 'Running image pre-processing...',
        uploadStageModel: 'Running medical model inference...',
        uploadStageFinal: 'Preparing final report...',
        uploadStageAnalyze: 'Analyzing image...',
        uploadPreparingResult: 'Preparing final result...',
        uploadDone: 'Analysis completed successfully.',
        uploadAnalyzeFailed: 'Analysis failed: {error}',
        unknownError: 'Unknown error',
        errorPrefix: 'Error: {error}',
        prescriptionElapsedSec: ' ({seconds}s)',
        prescriptionDefaultStatus: 'Processing...',
        prescriptionErrorStatus: 'An error occurred while analyzing the prescription{elapsed}',
        modelResultHeading: '🧠 Model analysis result:',
        modelResultAlt: 'Detection result',
        gradcamHeading: '📊 Focus map (Grad-CAM):',
        gradcamAlt: 'Grad-CAM visualization',
        gradcamHint: 'Red regions indicate areas the model focused on during diagnosis',
        chatErrorGeneric: 'Sorry, an error occurred. Please try again.',
        bookingTitle: 'Book an appointment',
        patientNameLabel: 'Name:',
        doctorLabel: 'Select doctor:',
        dayLabel: 'Select day:',
        timeLabel: 'Select time:',
        selectDoctorPlaceholder: '-- Select doctor --',
        selectDayPlaceholder: '-- Select day --',
        selectTimePlaceholder: '-- Select time --',
        bookingSubmit: 'Confirm booking',
        queueText: 'Your queue position: {queue} ({people} people before you)',
        fillAllFields: 'Please fill all fields',
        bookingErrorAlert: 'An error occurred while booking. Please try again.',
        bookingSuccessMessage: '✅ Your appointment has been booked successfully!\n\nDetails:\nDoctor: {doctor}\nDay: {date}\nTime: {time}\nName: {name}\n\nWe look forward to seeing you!',
        clearChatConfirm: 'Do you want to clear all messages in this chat?',
        clearChatSuccessToast: 'Chat history cleared',
        clearChatFailToast: 'Could not clear chat history',
        deleteChatConfirm: 'Do you want to delete this chat?',
        deleteChatToast: 'Chat deleted',
        deleteLabel: 'Delete',
        exportYouLabel: 'You',
        exportDoneToast: 'Chat exported',
        copyButtonLabel: 'Copy',
        copyDoneToast: 'Copied ✓',
        copyFailToast: 'Copy failed',
        imageTypeDentalPhotoLabel: '🦷 Dental photo',
        imageTypeDentalXrayLabel: '🩻 Dental X-ray',
        imageTypeBoneLabel: '🦴 Bone X-ray',
        imageTypeChestLabel: '🫁 Chest X-ray',
        imageTypePrescriptionLabel: '📝 Prescription',
        dayNames: {
            monday: 'Monday',
            tuesday: 'Tuesday',
            wednesday: 'Wednesday',
            thursday: 'Thursday',
            friday: 'Friday',
            saturday: 'Saturday',
            sunday: 'Sunday'
        }
    }
};

function interpolate(template, params = {}) {
    return String(template || '').replace(/\{(\w+)\}/g, (_, key) => {
        if (Object.prototype.hasOwnProperty.call(params, key)) {
            return String(params[key]);
        }
        return `{${key}}`;
    });
}

function t(key, params = {}) {
    const lang = getPreferredLanguage();
    const pack = I18N[lang] || I18N.ar;
    const parts = String(key || '').split('.');
    let value = pack;

    for (const part of parts) {
        if (value && Object.prototype.hasOwnProperty.call(value, part)) {
            value = value[part];
        } else {
            value = null;
            break;
        }
    }

    if (value == null) {
        return key;
    }

    return typeof value === 'string' ? interpolate(value, params) : value;
}

function getLanguageLocale() {
    return t('locale');
}

function getLocalizedDayName(dayKey) {
    const dayNames = t('dayNames');
    if (!dayKey || typeof dayNames !== 'object') return dayKey || '';
    const normalized = String(dayKey).trim().toLowerCase();
    return dayNames[normalized] || dayKey;
}

function getLocalizedUploadTypeLabel(type) {
    const labels = t('uploadTypeLabels');
    if (!labels || typeof labels !== 'object') return type;
    return labels[type] || type;
}


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
        doctorSelect.innerHTML = `<option value="">${t('selectDoctorPlaceholder')}</option>`;
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
        input.placeholder = message || t('analyzingPlaceholder');
    } else {
        input.disabled = false;
        sendButton.disabled = false;
        fileInput.disabled = false;
        sendButton.textContent = '➤';
        sendButton.style.opacity = '1';
        input.placeholder = t('messagePlaceholder');
        input.focus();
    }
}

// ==================== FILE UPLOAD FUNCTIONS ====================

function handleFileSelect(event) {
    if (isWaitingForResponse) {
        event.target.value = '';
        showToast(t('uploadWaitCurrentToast'));
        return;
    }

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

    const shouldAutoUpload = uploadPickerMode === 'auto';
    // Reset mode after one file selection to avoid accidental future auto-uploads.
    uploadPickerMode = 'manual';

    if (!shouldAutoUpload) {
        return;
    }

    // Auto-send selected file immediately (no need to press Send).
    // If the user typed a note, attach it to the same upload request.
    const input = document.getElementById('messageInput');
    const attachedMessage = input ? input.value.trim() : '';
    if (attachedMessage) {
        addMessage(attachedMessage, 'user');
    }
    if (input) {
        input.value = '';
    }

    uploadFile(attachedMessage);
}

function clearUpload() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('uploadPreview').classList.add('hidden');
    document.getElementById('imagePreview').classList.add('hidden');
    document.getElementById('pdfPreview').classList.add('hidden');
}

function setSelectedUploadType(type) {
    const typeInput = document.querySelector(`input[name="imageType"][value="${type}"]`);
    if (typeInput) typeInput.checked = true;
}

function prepareManualUploadContext() {
    uploadPickerMode = 'manual';
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.accept = DEFAULT_FILE_ACCEPT;
    }
}

function openManualUploadPicker() {
    prepareManualUploadContext();
    const fileInput = document.getElementById('fileInput');
    if (!fileInput) return;
    fileInput.click();
}

function openUploadPickerByType(type) {
    if (type === 'manual') {
        openManualUploadPicker();
        return;
    }

    const fileInput = document.getElementById('fileInput');
    if (!fileInput) return;

    if (type !== 'pdf_report') {
        setSelectedUploadType(type);
    }

    uploadPickerMode = 'auto';
    fileInput.accept = UPLOAD_TYPE_ACCEPT[type] || DEFAULT_FILE_ACCEPT;
    fileInput.click();
}

function inferUploadTypeFromMessage(message) {
    if (!message) return null;

    const text = String(message).toLowerCase().trim();
    if (!text) return null;

    const uploadIntentHints = [
        'ارفع', 'رفع', 'ابعت', 'ابعث', 'send', 'upload', 'share'
    ];

    const hasUploadIntent = uploadIntentHints.some((hint) => text.includes(hint));
    if (!hasUploadIntent) return null;

    if (text.includes('روشت') || text.includes('prescription') || text.includes('rx')) {
        return 'prescription';
    }

    if (text.includes('pdf') || text.includes('تقرير') || text.includes('report')) {
        return 'pdf_report';
    }

    if (text.includes('عظام') || text.includes('bone') || text.includes('wrist')) {
        return 'bone';
    }

    if (text.includes('صدر') || text.includes('chest') || text.includes('lung') || text.includes('رئة')) {
        return 'chest';
    }

    const hasDental = text.includes('اسنان') || text.includes('أسنان') || text.includes('dental') || text.includes('tooth');
    if (hasDental) {
        const xrayHints = ['اشعة', 'أشعة', 'اشعه', 'xray', 'x-ray', 'scan', 'panoramic'];
        const photoHints = ['صورة', 'صور', 'photo', 'camera'];

        if (xrayHints.some((hint) => text.includes(hint))) return 'dental_xray';
        if (photoHints.some((hint) => text.includes(hint))) return 'dental_photo';

        // Default dental intent to x-ray unless explicitly photo.
        return 'dental_xray';
    }

    return null;
}

function showUploadTypeOptionsInChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;

    const options = [
        { type: 'dental_xray', label: getLocalizedUploadTypeLabel('dental_xray') },
        { type: 'bone', label: getLocalizedUploadTypeLabel('bone') },
        { type: 'chest', label: getLocalizedUploadTypeLabel('chest') },
        { type: 'dental_photo', label: getLocalizedUploadTypeLabel('dental_photo') },
        { type: 'prescription', label: getLocalizedUploadTypeLabel('prescription') },
        { type: 'pdf_report', label: getLocalizedUploadTypeLabel('pdf_report') },
        { type: 'manual', label: getLocalizedUploadTypeLabel('manual') }
    ];

    const bubble = document.createElement('div');
    bubble.className = 'message bot-message upload-type-options-message';

    const card = document.createElement('div');
    card.className = 'upload-type-options-card';

    const title = document.createElement('div');
    title.className = 'upload-type-options-title';
    title.textContent = t('uploadTypeTitle');

    const grid = document.createElement('div');
    grid.className = 'upload-type-options-grid';

    options.forEach((option) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'upload-type-option-btn';
        btn.textContent = option.label;
        btn.addEventListener('click', () => openUploadPickerByType(option.type));
        grid.appendChild(btn);
    });

    card.appendChild(title);
    card.appendChild(grid);
    bubble.appendChild(card);
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createUploadProgressBubble(fileType) {
    const chatMessages = document.getElementById('chatMessages');
    const bubble = document.createElement('div');
    bubble.className = 'message bot-message upload-progress-message';

    const card = document.createElement('div');
    card.className = 'upload-progress-card';

    const header = document.createElement('div');
    header.className = 'upload-progress-header';

    const title = document.createElement('span');
    title.className = 'upload-progress-title';
    title.textContent = t('uploadProgressTitle', { fileType });

    const percent = document.createElement('span');
    percent.className = 'upload-progress-percent';
    percent.textContent = '0%';

    header.appendChild(title);
    header.appendChild(percent);

    const track = document.createElement('div');
    track.className = 'upload-progress-track';
    const fill = document.createElement('div');
    fill.className = 'upload-progress-fill';
    track.appendChild(fill);

    const status = document.createElement('div');
    status.className = 'upload-progress-status';
    status.textContent = t('uploadProgressStart');

    card.appendChild(header);
    card.appendChild(track);
    card.appendChild(status);
    bubble.appendChild(card);
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return {
        bubble,
        card,
        percent,
        fill,
        status,
        progress: 0,
    };
}

function updateUploadProgressBubble(ui, percent, statusText = null, isError = false) {
    if (!ui) return;
    const clamped = Math.max(0, Math.min(100, Math.round(percent)));
    ui.progress = Math.max(ui.progress || 0, clamped);
    ui.fill.style.width = `${ui.progress}%`;
    ui.percent.textContent = `${ui.progress}%`;
    if (statusText) {
        ui.status.textContent = statusText;
    }
    ui.card.classList.toggle('error', isError);
    if (!isError && ui.progress >= 100) {
        ui.card.classList.add('done');
    }
}

async function uploadFile(attachedMessage = '') {
    const fileToUpload = selectedFile;
    if (!fileToUpload) return null;

    const formData = new FormData();
    formData.append('file', fileToUpload);
    formData.append('patient_id', patientId);
    formData.append('language_preference', getPreferredLanguage());
    if (attachedMessage && attachedMessage.trim()) {
        formData.append('user_message', attachedMessage.trim());
    }

    // Get selected image type (dental, bone, or chest)
    const imageTypeInput = document.querySelector('input[name="imageType"]:checked');
    const imageType = imageTypeInput ? imageTypeInput.value : 'dental_xray';
    formData.append('image_type', imageType);

    // Show uploading message with correct label and image preview
    const fileType = fileToUpload.type.startsWith('image/')
        ? getLocalizedUploadTypeLabel(imageType || 'dental_xray')
        : getLocalizedUploadTypeLabel('pdf_report');

    // Clear the attachment preview immediately after sending so the same file
    // does not remain selected for the next message.
    clearUpload();

    // Show uploaded image preview in chat
    const chatMessages = document.getElementById('chatMessages');
    if (fileToUpload.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const userImgDiv = document.createElement('div');
            userImgDiv.className = 'message user-message';
            userImgDiv.innerHTML = `
                <p style="margin-bottom: 6px;">${t('uploadReceivedTag', { fileType })}</p>
                <img src="${e.target.result}" alt="${t('imagePreviewAlt')}"
                     style="max-width: 220px; max-height: 220px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.3); object-fit: cover; cursor: pointer;"
                     onclick="window.open(this.src, '_blank')" title="${t('clickToZoomTitle')}" />
            `;
            chatMessages.appendChild(userImgDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        };
        reader.readAsDataURL(fileToUpload);
    } else {
        addMessage(t('uploadReceivedTag', { fileType }), 'user');
    }

    const progressUI = createUploadProgressBubble(fileType);
    updateUploadProgressBubble(progressUI, 6, t('uploadProgressPreparing'));

    setLoadingState(true, t('uploadAnalyzingFile'));

    const inputEl = document.getElementById('messageInput');
    const setProgress = (percent, statusText, isError = false) => {
        updateUploadProgressBubble(progressUI, percent, statusText, isError);
        if (inputEl && statusText) {
            inputEl.placeholder = `⏳ ${statusText}`;
        }
    };

    let fallbackTicker = null;
    const startGenericProgress = () => {
        fallbackTicker = setInterval(() => {
            if (!progressUI || progressUI.progress >= 92) return;
            const current = progressUI.progress;
            const step = current < 30 ? 6 : current < 60 ? 4 : 2;
            const next = Math.min(92, current + step);

            let stageMsg = t('uploadStageAnalyze');
            if (next < 25) stageMsg = t('uploadStageUpload');
            else if (next < 55) stageMsg = t('uploadStagePreprocess');
            else if (next < 82) stageMsg = t('uploadStageModel');
            else stageMsg = t('uploadStageFinal');

            setProgress(next, stageMsg);
        }, 1200);
    };

    startGenericProgress();

    // For prescriptions, the VLM may need to lazy-load weights — poll the
    // backend status and reflect it live in the input placeholder so the user
    // knows the system is alive.
    let statusPoller = null;
    if (imageType === 'prescription') {
        const stageToPercent = {
            idle: 8,
            loading_drugs: 20,
            drugs_ready: 32,
            loading_model: 48,
            loading_processor: 58,
            downloading_weights: 76,
            model_ready: 88,
            inference: 93,
            post_processing: 97,
            done: 100,
        };

        const pollOnce = async () => {
            try {
                const r = await fetch('/api/prescription/status');
                if (!r.ok) return;
                const s = await r.json();
                const elapsed = s.elapsed_sec ? t('prescriptionElapsedSec', { seconds: Math.round(s.elapsed_sec) }) : '';
                const msg = s.message || s.stage || t('prescriptionDefaultStatus');
                const mappedPercent = stageToPercent[s.stage] || progressUI.progress;
                if (s.stage === 'error') {
                    setProgress(Math.max(progressUI.progress, 95), t('prescriptionErrorStatus', { elapsed }), true);
                    return;
                }
                setProgress(mappedPercent, `📝 ${msg}${elapsed}`);
            } catch (_) { /* ignore */ }
        };
        pollOnce();
        statusPoller = setInterval(pollOnce, 2000);
    }

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        setProgress(98, t('uploadPreparingResult'));

        if (data.success) {
            const chatMessages = document.getElementById('chatMessages');

            if (data.result_image) {
                const imageDiv = document.createElement('div');
                imageDiv.className = 'message bot-message';
                imageDiv.innerHTML = `
                    <p style="margin-bottom: 8px; font-weight: bold;">${t('modelResultHeading')}</p>
                    <img src="data:image/jpeg;base64,${data.result_image}"
                         alt="${t('modelResultAlt')}"
                        style="max-width: 560px; max-height: 560px; border-radius: 8px; border: 2px solid #0f766e; object-fit: contain; cursor: pointer;"
                         onclick="window.open(this.src, '_blank')" title="${t('clickToZoomTitle')}" />
                `;
                chatMessages.appendChild(imageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            if (data.image_type === 'chest' && data.gradcam_image) {
                const imageDiv = document.createElement('div');
                imageDiv.className = 'message bot-message gradcam-container';
                imageDiv.innerHTML = `
                    <p style="margin-bottom: 8px; font-weight: bold;">${t('gradcamHeading')}</p>
                    <img src="data:image/jpeg;base64,${data.gradcam_image}"
                         alt="${t('gradcamAlt')}"
                        style="max-width: 560px; max-height: 560px; border-radius: 8px; border: 2px solid #4a90a4; object-fit: contain; cursor: pointer;"
                         onclick="window.open(this.src, '_blank')" title="${t('clickToZoomTitle')}" />
                    <p style="margin-top: 8px; font-size: 12px; color: #888;">
                        ${t('gradcamHint')}
                    </p>
                `;
                chatMessages.appendChild(imageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            addMessage(data.report, 'bot');
            setProgress(100, t('uploadDone'));
        } else {
            const errorText = data.error || t('unknownError');
            setProgress(100, t('uploadAnalyzeFailed', { error: errorText }), true);
            addMessage(t('errorPrefix', { error: errorText }), 'bot');
        }

        return data;
    } catch (error) {
        console.error('Upload error:', error);
        setProgress(100, t('uploadReadError'), true);
        addMessage(t('uploadErrorGeneric'), 'bot');
        return null;
    } finally {
        if (statusPoller) clearInterval(statusPoller);
        if (fallbackTicker) clearInterval(fallbackTicker);
        clearUpload();
        setLoadingState(false);
    }
}

// ==================== MESSAGE FUNCTIONS ====================

function sendMessage(event) {
    event.preventDefault();

    if (isWaitingForResponse) return;

    const input = document.getElementById('messageInput');
    const message = input.value.trim();

    if (!selectedFile && message) {
        const inferredUploadType = inferUploadTypeFromMessage(message);
        if (inferredUploadType) {
            openUploadPickerByType(inferredUploadType);
            showToast(t('uploadPickNowToast'));
            return;
        }
    }

    // Allow sending if there's a message OR a file selected
    if (!message && !selectedFile) return;


    // Check if there's a file to upload
    if (selectedFile) {
        // If user also typed a message, show it
        if (message) {
            addMessage(message, 'user');
        }
        input.value = '';
        uploadFile(message);
        return;
    }

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
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    const p = document.createElement('p');
    messageDiv.appendChild(p);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: userMessage,
                patient_id: patientId,
                language_preference: getPreferredLanguage()
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const contentType = response.headers.get('content-type') || '';
        let fullText = '';
        let queryType = '';

        if (response.body && contentType.includes('application/x-ndjson')) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    let payload;
                    try {
                        payload = JSON.parse(line);
                    } catch (parseError) {
                        continue;
                    }

                    if (payload.chunk) {
                        fullText += payload.chunk;
                        p.innerHTML = fullText.replace(/\n/g, '<br>');
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }

                    if (payload.done) {
                        if (payload.response) {
                            fullText = payload.response;
                            p.innerHTML = fullText.replace(/\n/g, '<br>');
                        }
                        queryType = payload.query_type || '';
                    }
                }
            }
        } else {
            const data = await response.json();
            fullText = data.response || data.message || '';
            p.innerHTML = fullText.replace(/\n/g, '<br>');
            queryType = data.query_type || '';
        }

        // Auto-open booking modal for appointment requests
        if (queryType === 'appointment') {
            setTimeout(() => openBookingModal(), 500);
        }

        if (queryType === 'upload') {
            setTimeout(() => showUploadTypeOptionsInChat(), 180);
        }

    } catch (error) {
        console.error('Error:', error);
        messageDiv.remove();
        addMessage(t('chatErrorGeneric'), 'bot');
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

function runQuickAction(actionKey) {
    const promptMap = {
        flu: t('quickPromptFlu'),
        appointment: t('quickPromptAppointment'),
        hours: t('quickPromptHours')
    };
    const prompt = promptMap[actionKey] || '';
    if (!prompt) return;
    quickAction(prompt);
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

    timeSelect.innerHTML = `<option value="">${t('selectTimePlaceholder')}</option>`;
    daySelect.innerHTML = `<option value="">${t('selectDayPlaceholder')}</option>`;

    if (!doctorId) return;

    selectedDoctor = doctors.find(d => d.id == doctorId);

    if (selectedDoctor) {
        const days = Object.keys(selectedDoctor.availability);
        days.forEach(day => {
            const option = document.createElement('option');
            option.value = day;
            option.textContent = getLocalizedDayName(day);
            daySelect.appendChild(option);
        });
    }
}

function updateTimeSlots() {
    const day = document.getElementById('daySelect').value;
    const timeSelect = document.getElementById('timeSelect');

    timeSelect.innerHTML = `<option value="">${t('selectTimePlaceholder')}</option>`;

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
        queueText.textContent = t('queueText', {
            queue: data.queue_position,
            people: data.people_before_you
        });
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
        alert(t('fillAllFields'));
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
            const confirmationMsg = t('bookingSuccessMessage', {
                doctor: doctor ? doctor.name : '-',
                date: getLocalizedDayName(date),
                time,
                name: patientName
            });

            addMessage(confirmationMsg, 'bot');
            closeBookingModal();

            document.getElementById('bookingForm').reset();
            document.getElementById('queueInfo').classList.add('hidden');
        } else {
            alert(t('bookingErrorAlert'));
        }
    } catch (error) {
        console.error('Error booking appointment:', error);
        alert(t('bookingErrorAlert'));
    }
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('bookingModal');
    if (event.target == modal) {
        closeBookingModal();
    }
}

/* =========================================================
 * Chat sessions, sidebar, theme, export, copy buttons
 * =======================================================*/

const CHATS_KEY = 'cliniq_chats';
const ACTIVE_CHAT_KEY = 'cliniq_active_chat';
const THEME_KEY = 'cliniq_theme';
const LANGUAGE_KEY = 'cliniq_default_language';

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = value;
    }
}

function setTitleAndAria(id, title, ariaLabel) {
    const el = document.getElementById(id);
    if (!el) return;
    if (title != null) {
        el.title = title;
    }
    if (ariaLabel != null) {
        el.setAttribute('aria-label', ariaLabel);
    }
}

function applyUILanguage(lang) {
    const normalized = String(lang || '').toLowerCase() === 'en' ? 'en' : 'ar';
    const pack = I18N[normalized] || I18N.ar;

    document.documentElement.lang = normalized;
    document.documentElement.dir = pack.dir || (normalized === 'en' ? 'ltr' : 'rtl');
    document.title = pack.pageTitle;

    setText('sidebarTitle', pack.sidebarTitle);
    setText('newChatLabel', pack.newChat);
    setText('headerSubtitle', pack.appSubtitle);
    setText('welcomeMessage', pack.welcomeMessage);

    setText('imageTypeDentalPhotoLabel', pack.imageTypeDentalPhotoLabel);
    setText('imageTypeDentalXrayLabel', pack.imageTypeDentalXrayLabel);
    setText('imageTypeBoneLabel', pack.imageTypeBoneLabel);
    setText('imageTypeChestLabel', pack.imageTypeChestLabel);
    setText('imageTypePrescriptionLabel', pack.imageTypePrescriptionLabel);

    setText('quickFluBtn', pack.quickFluLabel);
    setText('quickBookingBtn', pack.quickBookingLabel);
    setText('quickScanBtn', pack.quickScanLabel);
    setText('quickManualUploadBtn', pack.quickManualUploadLabel);
    setText('quickHoursBtn', pack.quickHoursLabel);

    setText('bookingModalTitle', pack.bookingTitle);
    setText('patientNameLabel', pack.patientNameLabel);
    setText('doctorSelectLabel', pack.doctorLabel);
    setText('daySelectLabel', pack.dayLabel);
    setText('timeSelectLabel', pack.timeLabel);
    setText('bookingSubmitBtn', pack.bookingSubmit);

    setTitleAndAria('toggleSidebarBtn', pack.toggleSidebarTitle, 'toggle sidebar');
    setTitleAndAria('sidebarOpenBtn', pack.openSidebarTitle, 'open sidebar');
    setTitleAndAria('languageSelect', pack.languageSelectTitle, 'default language');
    setTitleAndAria('clearCurrentChatBtn', pack.clearCurrentChatTitle, 'clear current chat history');
    setTitleAndAria('exportChatBtn', pack.exportChatTitle, 'export chat');
    setTitleAndAria('themeToggleBtn', pack.toggleThemeTitle, 'toggle theme');
    setTitleAndAria('manualUploadPinLabel', pack.pinUploadTitle, null);

    const input = document.getElementById('messageInput');
    if (input && !isWaitingForResponse) {
        input.placeholder = pack.messagePlaceholder;
    }

    const doctorSelect = document.getElementById('doctorSelect');
    if (doctorSelect) {
        const first = doctorSelect.querySelector('option[value=""]');
        if (first) first.textContent = pack.selectDoctorPlaceholder;
    }

    const daySelect = document.getElementById('daySelect');
    if (daySelect) {
        const first = daySelect.querySelector('option[value=""]');
        if (first) first.textContent = pack.selectDayPlaceholder;
    }

    const timeSelect = document.getElementById('timeSelect');
    if (timeSelect) {
        const first = timeSelect.querySelector('option[value=""]');
        if (first) first.textContent = pack.selectTimePlaceholder;
    }

    document.querySelectorAll('.msg-action-btn').forEach((btn) => {
        btn.textContent = pack.copyButtonLabel;
    });
}

function getPreferredLanguage() {
    const lang = String(localStorage.getItem(LANGUAGE_KEY) || 'ar').toLowerCase();
    return lang === 'en' ? 'en' : 'ar';
}

function setPreferredLanguage(lang) {
    const normalized = String(lang || '').toLowerCase() === 'en' ? 'en' : 'ar';
    localStorage.setItem(LANGUAGE_KEY, normalized);
    const select = document.getElementById('languageSelect');
    if (select && select.value !== normalized) {
        select.value = normalized;
    }
    applyUILanguage(normalized);
    renderChatList();
}

function loadChats() {
    try { return JSON.parse(localStorage.getItem(CHATS_KEY)) || []; }
    catch (e) { return []; }
}

function saveChats(chats) {
    localStorage.setItem(CHATS_KEY, JSON.stringify(chats));
}

function isDefaultChatTitle(title) {
    const normalized = String(title || '').trim();
    return normalized === '' || normalized === 'محادثة جديدة' || normalized === 'New chat';
}

function ensureActiveChat() {
    let chats = loadChats();
    let activeId = localStorage.getItem(ACTIVE_CHAT_KEY);

    // Migrate legacy single-session patient_id into a chat entry
    if (!chats.length) {
        const legacyId = localStorage.getItem('patient_id');
        const id = legacyId || ('patient_' + Math.random().toString(36).substr(2, 9));
        chats.push({ id, title: t('newChat'), updated: Date.now() });
        saveChats(chats);
        activeId = id;
        localStorage.setItem(ACTIVE_CHAT_KEY, activeId);
    }

    if (!activeId || !chats.find(c => c.id === activeId)) {
        activeId = chats[0].id;
        localStorage.setItem(ACTIVE_CHAT_KEY, activeId);
    }

    patientId = activeId;
    localStorage.setItem('patient_id', patientId);
    return activeId;
}

function renderChatList() {
    const chats = loadChats().sort((a, b) => (b.updated || 0) - (a.updated || 0));
    const list = document.getElementById('chatList');
    if (!list) return;
    const activeId = localStorage.getItem(ACTIVE_CHAT_KEY);

    if (!chats.length) {
        list.innerHTML = `<div class="chat-list-empty">${t('noChats')}</div>`;
        return;
    }

    list.innerHTML = '';
    chats.forEach(chat => {
        const item = document.createElement('div');
        item.className = 'chat-list-item' + (chat.id === activeId ? ' active' : '');
        item.dataset.id = chat.id;
        const dt = chat.updated ? new Date(chat.updated) : null;
        const meta = dt ? dt.toLocaleString(getLanguageLocale(), { dateStyle: 'short', timeStyle: 'short' }) : '';
        item.innerHTML = `
            <div class="chat-item-top">
                <span class="title"></span>
                <button class="delete-chat" title="${t('deleteLabel')}" aria-label="delete">🗑</button>
            </div>
            <span class="meta"></span>
        `;
        item.querySelector('.title').textContent = isDefaultChatTitle(chat.title)
            ? t('newChat')
            : (chat.title || t('chatFallbackTitle'));
        item.querySelector('.meta').textContent = meta;
        item.addEventListener('click', (e) => {
            if (e.target.closest('.delete-chat')) return;
            switchChat(chat.id);
        });
        item.querySelector('.delete-chat').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteChat(chat.id);
        });
        list.appendChild(item);
    });
}

function touchActiveChat(titleHint) {
    const chats = loadChats();
    const chat = chats.find(c => c.id === patientId);
    if (!chat) return;
    chat.updated = Date.now();
    if (titleHint && isDefaultChatTitle(chat.title)) {
        chat.title = titleHint.slice(0, 40);
    }
    saveChats(chats);
    renderChatList();
}

function newChat() {
    const id = 'patient_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
    const chats = loadChats();
    chats.push({ id, title: t('newChat'), updated: Date.now() });
    saveChats(chats);
    localStorage.setItem(ACTIVE_CHAT_KEY, id);
    patientId = id;
    localStorage.setItem('patient_id', patientId);
    resetChatUI();
    renderChatList();
}

function resetChatUI() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = `
        <div class="message bot-message">
            <p>${t('welcomeMessage')}</p>
        </div>`;
}

async function clearCurrentChatHistory() {
    if (!patientId) return;
    if (!confirm(t('clearChatConfirm'))) return;

    try {
        const res = await fetch('/api/chat/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patient_id: patientId })
        });

        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }

        resetChatUI();
        touchActiveChat();
        showToast(t('clearChatSuccessToast'));
    } catch (e) {
        console.error('clearCurrentChatHistory error:', e);
        showToast(t('clearChatFailToast'));
    }
}

async function switchChat(id) {
    if (!id || id === patientId) return;
    patientId = id;
    localStorage.setItem(ACTIVE_CHAT_KEY, id);
    localStorage.setItem('patient_id', id);
    resetChatUI();
    renderChatList();

    await loadChatHistory(id);
}

async function loadChatHistory(id) {
    if (!id) return;
    patientId = id;
    localStorage.setItem('patient_id', id);

    try {
        const res = await fetch(`/api/chat/history?patient_id=${encodeURIComponent(id)}`);
        if (!res.ok) return;
        const data = await res.json();
        const history = data.history || [];
        const chatMessages = document.getElementById('chatMessages');
        if (history.length) chatMessages.innerHTML = '';
        history.forEach(msg => {
            const sender = msg.role === 'user' ? 'user' : 'bot';
            addMessage(msg.content || msg.message || '', sender);
        });
    } catch (e) { console.error('switchChat error:', e); }
}

async function deleteChat(id) {
    if (!confirm(t('deleteChatConfirm'))) return;
    const deletedWasActive = (id === patientId);
    let chats = loadChats().filter(c => c.id !== id);

    if (deletedWasActive && chats.length) {
        const nextId = chats[0].id;
        patientId = nextId;
        localStorage.setItem(ACTIVE_CHAT_KEY, nextId);
        localStorage.setItem('patient_id', nextId);
    }

    saveChats(chats);
    renderChatList();

    try { await fetch('/api/chat/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: id })
    }); } catch (e) { /* ignore */ }

    if (deletedWasActive) {
        if (chats.length) {
            resetChatUI();
            await loadChatHistory(patientId);
        } else {
            newChat();
        }
    }

    showToast(t('deleteChatToast'));
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
}

/* ---------- theme ---------- */
function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const btn = document.getElementById('themeToggleBtn');
    if (btn) btn.textContent = theme === 'dark' ? '☀️' : '🌙';
}
function toggleTheme() {
    const current = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
}

/* ---------- export ---------- */
function exportChat() {
    const chatMessages = document.getElementById('chatMessages');
    const lines = [];
    chatMessages.querySelectorAll('.message').forEach(node => {
        const who = node.classList.contains('user-message') ? t('exportYouLabel') : 'ClinIQ';
        const text = (node.querySelector('p')?.innerText || '').trim();
        if (text) lines.push(`${who}:\n${text}\n`);
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cliniq-chat-${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.txt`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    showToast(t('exportDoneToast'));
}

/* ---------- toast ---------- */
function showToast(msg) {
    let el = document.querySelector('.toast');
    if (!el) {
        el = document.createElement('div');
        el.className = 'toast';
        document.body.appendChild(el);
    }
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => el.classList.remove('show'), 1800);
}

/* ---------- copy buttons on bot messages ---------- */
function attachBotActions(messageDiv) {
    if (!messageDiv || messageDiv.querySelector('.message-actions')) return;
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    const copyBtn = document.createElement('button');
    copyBtn.className = 'msg-action-btn';
    copyBtn.type = 'button';
    copyBtn.textContent = t('copyButtonLabel');
    copyBtn.onclick = () => {
        const txt = messageDiv.querySelector('p')?.innerText || '';
        navigator.clipboard.writeText(txt).then(
            () => showToast(t('copyDoneToast')),
            () => showToast(t('copyFailToast'))
        );
    };
    actions.appendChild(copyBtn);
    messageDiv.appendChild(actions);
}

// Wrap addMessage so every bot bubble gets a copy button.
const _origAddMessage = addMessage;
addMessage = function (text, sender) {
    _origAddMessage(text, sender);
    const chatMessages = document.getElementById('chatMessages');
    const last = chatMessages.lastElementChild;
    if (sender === 'bot' && last) attachBotActions(last);
    if (sender === 'user') {
        touchActiveChat(text);
    }
};

// Observe the bot streaming bubble (created directly in getResponse) and add actions when its text settles.
const _msgObserver = new MutationObserver(() => {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.querySelectorAll('.bot-message').forEach(div => {
        if (!div.querySelector('.message-actions') && div.querySelector('p')) {
            attachBotActions(div);
        }
    });
});

document.addEventListener('DOMContentLoaded', () => {
    // Theme
    const savedTheme = localStorage.getItem(THEME_KEY) || 'light';
    applyTheme(savedTheme);

    // Language fallback for no-text uploads
    setPreferredLanguage(getPreferredLanguage());

    // Chats
    const activeId = ensureActiveChat();
    renderChatList();
    loadChatHistory(activeId);

    // Mobile: collapse sidebar by default
    if (window.matchMedia('(max-width: 768px)').matches) {
        document.getElementById('sidebar')?.classList.add('collapsed');
    }

    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        _msgObserver.observe(chatMessages, { childList: true, subtree: true });
    }
});
