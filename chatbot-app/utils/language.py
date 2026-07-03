import re

def normalize_language_code(language):
    """Normalize UI/backend language codes to 'ar' or 'en'."""
    code = str(language or 'ar').strip().lower()
    if code.startswith('en'):
        return 'en'
    return 'ar'


def detect_text_language(text: str):
    """Lightweight language detection for Arabic/English user text."""
    content = str(text or '').strip()
    if not content:
        return None

    arabic_matches = re.findall(r'[\u0600-\u06FF]', content)
    english_matches = re.findall(r'[A-Za-z]', content)

    if arabic_matches and not english_matches:
        return 'ar'
    if english_matches and not arabic_matches:
        return 'en'
    if arabic_matches and english_matches:
        return 'ar' if len(arabic_matches) >= len(english_matches) else 'en'

    return None


def resolve_response_language(user_text: str = None, preferred_language: str = 'ar'):
    """Use message language when available; otherwise fallback to user preference."""
    detected = detect_text_language(user_text)
    if detected:
        return detected
    return normalize_language_code(preferred_language)


def get_language_policy(target_language: str = 'ar'):
    """Return strict language policy text for prompts."""
    code = normalize_language_code(target_language)
    if code == 'en':
        return (
            "Language policy (strict): Respond in English only. "
            "Do not include Arabic words, Arabic sentences, or bilingual duplicates."
        )
    return (
        "سياسة اللغة (إلزامي): أجب بالعربية فقط. "
        "لا تضف ترجمة إنجليزية ولا تكتب ردًا ثنائي اللغة."
    )


def build_upload_guidance_message(language: str = 'ar'):
    """Return upload guidance in Arabic or English."""
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
