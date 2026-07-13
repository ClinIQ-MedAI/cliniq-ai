from core.settings import *


def load_json(filepath):
    # Load JSON file.
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return []


def save_json(filepath, data):
    # Save JSON file.
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def clean_markdown(text: str):
    # Remove markdown formatting from text for JSON compatibility.
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


def normalize_language_code(language):
    # Normalize UI/backend language codes to 'ar' or 'en'.
    code = str(language or 'ar').strip().lower()
    if code.startswith('en'):
        return 'en'
    return 'ar'


def detect_text_language(text: str):
    # Lightweight language detection for Arabic/English user text.
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
    # Use message language when available; otherwise fallback to user preference.
    detected = detect_text_language(user_text)
    if detected:
        return detected
    return normalize_language_code(preferred_language)


def get_language_policy(target_language: str = 'ar'):
    # Return strict language policy text for prompts.
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


def allowed_file(filename, allowed_extensions):
    # Check if file extension is allowed.
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def confidence_to_percent(value):
    # Convert confidence values to percentage (0-100).
    try:
        if isinstance(value, str):
            numeric = float(value.strip().replace('%', ''))
        else:
            numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric <= 1.0:
        numeric *= 100.0

    return max(0.0, min(100.0, numeric))

