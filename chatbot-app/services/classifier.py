import re

INTENT_PATTERNS = {
    "upload": {
        "keywords": ["رفع", "صور", "صورة", "أشعة", "اشعه", "upload", "image", "xray", "x-ray", "pdf", "تقرير", "scan", "ارفع", "صوره", "روشتة", "روشته", "prescription"],
        "weight": 1.0
    },
    "appointment": {
        "keywords": ["book", "appointment", "schedule", "reserve", "حجز", "موعد", "دكتور", "طبيب", "عيادة", "كشف", "ميعاد", "احجز", "احدد"],
        "weight": 1.2
    },
    "availability": {
        "keywords": ["available", "doctor", "when", "slot", "متاح", "مواعيد", "متى", "امتى", "ساعة", "ساعات", "ايام"],
        "weight": 1.0
    },
    "faq": {
        "keywords": ["how to", "insurance", "address", "location", "تأمين", "عنوان", "مكان", "اسئلة", "سؤال", "مين", "بكام", "تكلفة"],
        "weight": 0.8
    },
    "health": {
        "keywords": ["عندي", "وجع", "ألم", "الم", "علاج", "مرض", "pain", "symptoms", "sick", "cure", "تعبان", "حاسس", "صداع"],
        "weight": 0.9
    }
}

def classify_query(user_message):
    """
    Classify query intent using a weighted multi-signal scorer.
    """
    if not user_message:
        return "health"
        
    msg_lower = user_message.lower()
    scores = {intent: 0.0 for intent in INTENT_PATTERNS}
    
    # Tokenize (simple whitespace split)
    tokens = re.split(r'\s+', msg_lower)
    
    for intent, data in INTENT_PATTERNS.items():
        weight = data["weight"]
        matches = 0
        for kw in data["keywords"]:
            if kw in msg_lower:
                matches += 1
                # Exact word matches get a slight bonus
                if kw in tokens:
                    matches += 0.5
        scores[intent] = matches * weight
        
    # Penalty for conflicting intents (if upload and appointment both have high scores, adjust)
    # Just return the one with the highest score, unless all are 0
    best_intent = max(scores.items(), key=lambda x: x[1])
    
    if best_intent[1] >= 0.8:
        return best_intent[0]
        
    return "health"  # Default fallback
