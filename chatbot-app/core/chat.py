from core.settings import *
from core.textutil import *
from core.llm import *
# `import *` skips underscore names, so pull the RAG hook in explicitly (same
# reason as _screen_upload in app.py). Without this it's undefined here and every
# grounded answer silently falls back to "grounding skipped".
from core.settings import _rag_grounding


def build_upload_guidance_message(language: str = 'ar'):
    # Return upload guidance in Arabic or English.
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


def classify_query(user_message: str):
    # Classify the user's query type.
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
    # Handle general health questions using LLM.
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

    # RAG: ground the answer in the curated medical knowledge base when relevant.
    if RAG_SUPPORT:
        try:
            block = _rag_grounding(user_message, language="ar" if target_language == "ar" else "en")
            if block:
                system_message += f"\n\n{block}"
        except Exception as _e:  # noqa: BLE001 - never let RAG break the chat
            print(f"[rag] grounding skipped: {_e}")

    response = get_llm_response(user_message, system_message, history)
    return response


def handle_appointment_request(user_message: str, history: list = None, target_language: str = 'ar'):
    # Handle appointment booking requests.
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
    # Handle doctor availability queries.
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
    # Handle FAQ queries.
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

