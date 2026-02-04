"""
Regex Extraction Module
Rule-based extraction of drug name, dosage, and frequency from prescription text.
Improved for Arabic prescriptions.
"""
import re
from typing import Optional, List
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FILTER LISTS - Words to exclude from drug detection
# =============================================================================

# Words that are NOT drug names (doctor names, categories, clinic info)
EXCLUDE_WORDS = {
    # English category labels (not drug names)
    "lubricant", "analgesic", "antibiotic", "antiseptic", "antifungal",
    "anti-inflammatory", "antihistamine", "antacid", "laxative",
    "ointment", "cream", "gel", "drops", "syrup", "tablet", "capsule",
    "injection", "inhaler", "spray", "patch", "suppository",
    
    # Common English words on prescriptions
    "doctor", "dr", "patient", "name", "date", "age", "hospital",
    "clinic", "pharmacy", "prescription", "signature", "refill",
    "tab", "cap", "inj", "susp", "sol", "syr",
    "moaz", "ateto", "mohamed", "ahmed", "ali", "omar", "hassan",  # Common names
    "ey", "eye", "ear", "nose", "skin", "oral",  # Body parts
    "antibiotic ey", "antibiotic eye",  # Combined patterns
    
    # Arabic category/form words
    "قطرة", "مرهم", "كريم", "حبوب", "شراب", "حقن", "بخاخ",
    "دكتور", "مريض", "مستشفى", "عيادة", "صيدلية", "روشتة",
    "طبيب", "اخصائي", "استشاري", "جراح",
}

# Arabic dosage forms with English translations
ARABIC_FORMS = {
    "قطرة": "drops",
    "نقط": "drops", 
    "مرهم": "ointment",
    "كريم": "cream",
    "جل": "gel",
    "حبوب": "tablets",
    "أقراص": "tablets",
    "قرص": "tablet",
    "صرق": "tablet",  # OCR misread
    "كبسول": "capsule",
    "كبسولات": "capsules",
    "شراب": "syrup",
    "محلول": "solution",
    "حقن": "injection",
    "بخاخ": "inhaler/spray",
    "لبوس": "suppository",
}

# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Dosage patterns: captures amount and unit (mg, g, ml, etc.)
DOSAGE_PATTERNS = [
    r'(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|iu|units?)',  # 500mg, 1.5g, 10ml
    r'(\d+(?:\.\d+)?)\s*(مجم|جم|مل)',  # Arabic: مجم=mg, جم=g, مل=ml
    r'(\d+)\s*/\s*(\d+)\s*(mg|ml)',  # 500/125mg (combination drugs)
    r'(\d+)\s*%',  # Percentage (for ointments)
]

# Frequency patterns - Arabic improved
FREQUENCY_PATTERNS = [
    # English patterns
    (r'once\s+daily|once\s+a\s+day|od|qd', '1 time per day'),
    (r'twice\s+daily|twice\s+a\s+day|bd|bid', '2 times per day'),
    (r'three\s+times?\s+(?:a\s+)?day|tid|tds', '3 times per day'),
    (r'four\s+times?\s+(?:a\s+)?day|qid|qds', '4 times per day'),
    (r'every\s+(\d+)\s+hours?', r'every \1 hours'),
    (r'(\d+)\s*x\s*(\d+)', r'\2 times per day'),  # 1x3 format
    (r'e\.?d\.?', 'every day'),  # E.D. = every day
    
    # Arabic patterns - more comprehensive
    (r'مرة\s*واحدة\s*يوميا?', '1 time per day'),
    (r'مرة\s*يوميا?', '1 time per day'),
    (r'مرتين\s*يوميا?', '2 times per day'),
    (r'نيترم', '2 times per day'),  # OCR misread of مرتين
    (r'ثلاث\s*مرات\s*يوميا?', '3 times per day'),
    (r'٣\s*مرات', '3 times per day'),
    (r'يوميا', 'daily'),
    (r'ًايموي', 'daily'),  # OCR misread
    (r'صباحا?\s*و?\s*مساء?', 'morning and evening'),
    (r'قبل\s*النوم', 'before sleep'),
    (r'بعد\s*الاكل', 'after meals'),
    (r'قبل\s*الاكل', 'before meals'),
    (r'عند\s*اللزوم', 'as needed'),
    (r'كل\s*(\d+)\s*ساعات?', r'every \1 hours'),
]

# Duration patterns - Arabic improved
DURATION_PATTERNS = [
    (r'for\s+(\d+)\s+days?', r'\1 days'),
    (r'for\s+(\d+)\s+weeks?', r'\1 weeks'),
    (r'لمدة\s*(\d+)\s*(?:يوم|أيام)', r'\1 days'),
    (r'لمدة\s*(\d+)\s*(?:أسبوع|اسابيع)', r'\1 weeks'),
    (r'(\d+)\s*(?:يوم|أيام)', r'\1 days'),
    (r'أسبوع|اسبوع', '1 week'),
    (r'أسبوعين|اسبوعين', '2 weeks'),
    (r'نيعوبسا', '2 weeks'),  # OCR misread
]

# Route of administration
ROUTE_PATTERNS = [
    (r'oral|by\s+mouth|po', 'oral'),
    (r'iv|intravenous', 'intravenous'),
    (r'im|intramuscular', 'intramuscular'),
    (r'topical|apply\s+to', 'topical'),
    (r'eye|ophthalmic', 'ophthalmic'),
    (r'ear|otic', 'otic'),
    (r'عن\s+طريق\s+الفم', 'oral'),
    (r'للعين|عيون', 'ophthalmic'),
    (r'للاذن', 'otic'),
    (r'موضعي', 'topical'),
]


def is_valid_drug_candidate(text: str) -> bool:
    """
    Check if text is a valid drug name candidate.
    Filters out category labels, doctor names, etc.
    """
    text_lower = text.lower().strip()
    
    # Too short
    if len(text_lower) < 3:
        return False
    
    # Is it in exclude list?
    if text_lower in EXCLUDE_WORDS:
        return False
    
    # Starts with common non-drug patterns
    if re.match(r'^(dr\.?|d\.|دكتور|طبيب)', text_lower, re.IGNORECASE):
        return False
    
    # Is it a number only?
    if re.match(r'^\d+$', text_lower):
        return False
    
    # Check if ANY word is in exclude list
    words = text_lower.split()
    for word in words:
        if word in EXCLUDE_WORDS:
            return False
    
    # Check if it looks like a human name (two capitalized words, no numbers)
    if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', text) and not any(c.isdigit() for c in text):
        # Likely a human name like "Moaz Ateto"
        return False
    
    return True


def extract_dosage_form(text: str) -> Optional[str]:
    """
    Extract the dosage form (tablet, drops, ointment, etc.)
    """
    text_lower = text.lower()
    
    # English forms
    english_forms = {
        'tab': 'tablet', 'tablet': 'tablet', 'tablets': 'tablets',
        'cap': 'capsule', 'capsule': 'capsule', 'capsules': 'capsules',
        'drops': 'drops', 'drop': 'drops',
        'ointment': 'ointment', 'cream': 'cream', 'gel': 'gel',
        'syrup': 'syrup', 'suspension': 'suspension',
        'injection': 'injection', 'inj': 'injection',
        'inhaler': 'inhaler', 'spray': 'spray',
    }
    
    for key, value in english_forms.items():
        if key in text_lower:
            return value
    
    # Arabic forms
    for arabic, english in ARABIC_FORMS.items():
        if arabic in text:
            return english
    
    return None


def extract_dosage(text: str) -> Optional[dict]:
    """
    Extract dosage information from text.
    """
    text_lower = text.lower()
    
    for pattern in DOSAGE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return {
                    "amount": groups[0],
                    "unit": groups[1],
                    "raw": match.group(0)
                }
            elif len(groups) == 3:
                return {
                    "amount": f"{groups[0]}/{groups[1]}",
                    "unit": groups[2],
                    "raw": match.group(0)
                }
    
    return None


def extract_frequency(text: str) -> Optional[str]:
    """
    Extract frequency/dosing schedule from text.
    """
    for pattern, replacement in FREQUENCY_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if '\\' in replacement:
                return re.sub(pattern, replacement, match.group(0), flags=re.IGNORECASE)
            return replacement
    
    return None


def extract_duration(text: str) -> Optional[str]:
    """
    Extract treatment duration from text.
    """
    for pattern, replacement in DURATION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if '\\' in replacement:
                return re.sub(pattern, replacement, match.group(0), flags=re.IGNORECASE)
            return replacement
    
    return None


def extract_route(text: str) -> Optional[str]:
    """
    Extract route of administration from text.
    """
    for pattern, route in ROUTE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return route
    
    return None


def extract_drug_name(text: str, known_drugs: list = None) -> Optional[str]:
    """
    Extract potential drug name from text.
    Filters out non-drug words.
    """
    # Common drug name patterns (capitalized words)
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    
    matches = re.findall(pattern, text)
    
    for match in matches:
        if is_valid_drug_candidate(match):
            return match
    
    return None


def extract_drug_info(text: str) -> dict:
    """
    Extract all drug information from prescription text.
    """
    result = {
        "drug": None,
        "dosage": None,
        "frequency": None,
        "duration": None,
        "route": None,
        "form": None,
        "raw_text": text
    }
    
    # Extract drug name (filter out non-drugs)
    drug_name = extract_drug_name(text)
    if drug_name and is_valid_drug_candidate(drug_name):
        result["drug"] = drug_name
    
    # Extract dosage form
    form = extract_dosage_form(text)
    if form:
        result["form"] = form
    
    # Extract dosage
    dosage = extract_dosage(text)
    if dosage:
        result["dosage"] = f"{dosage['amount']}{dosage['unit']}"
    
    # Extract frequency
    frequency = extract_frequency(text)
    if frequency:
        result["frequency"] = frequency
    
    # Extract duration
    duration = extract_duration(text)
    if duration:
        result["duration"] = duration
    
    # Extract route
    route = extract_route(text)
    if route:
        result["route"] = route
    
    return result


def extract_prescription_blocks(text: str) -> List[dict]:
    """
    Extract medication blocks from prescription.
    Groups related lines together based on medication markers.
    
    Args:
        text: Full prescription OCR text
        
    Returns:
        List of medication info dictionaries
    """
    results = []
    
    # Split by newlines
    lines = text.strip().split('\n')
    
    # Medication markers
    med_markers = ['Lubricant', 'Analgesic', 'Antibiotic', 'Antifungal', 
                   'Anti-inflammatory', 'Antihistamine', 'قطرة', 'مرهم', 
                   'كريم', 'حبوب', 'شراب', 'TAB', 'CAP', 'E.D.', 'Ointment']
    
    current_block = []
    current_category = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this line is a medication marker
        is_marker = False
        for marker in med_markers:
            if marker.lower() in line.lower():
                is_marker = True
                # Save previous block if exists
                if current_block:
                    block_text = ' '.join(current_block)
                    info = extract_drug_info(block_text)
                    if current_category:
                        info['category'] = current_category
                    if info.get('drug') or info.get('form') or info.get('dosage') or info.get('frequency'):
                        results.append(info)
                
                # Start new block
                current_block = [line]
                current_category = marker
                break
        
        if not is_marker:
            current_block.append(line)
    
    # Save last block
    if current_block:
        block_text = ' '.join(current_block)
        info = extract_drug_info(block_text)
        if current_category:
            info['category'] = current_category
        if info.get('drug') or info.get('form') or info.get('dosage') or info.get('frequency'):
            results.append(info)
    
    return results


def extract_multiple_drugs(text: str) -> list:
    """
    Extract information for multiple drugs from a prescription.
    Uses block-based extraction for better results.
    """
    # Try block-based extraction first
    results = extract_prescription_blocks(text)
    
    if results:
        return results
    
    # Fallback: line-by-line extraction
    lines = re.split(r'\n|(?:\d+\.\s+)', text)
    
    results = []
    for line in lines:
        line = line.strip()
        if len(line) > 3:
            info = extract_drug_info(line)
            # Only include if we found meaningful info
            if info.get("drug") or info.get("dosage") or info.get("frequency") or info.get("form"):
                # Extra filter: skip if drug name is not valid
                if info.get("drug") and not is_valid_drug_candidate(info["drug"]):
                    info["drug"] = None
                results.append(info)
    
    return results


if __name__ == "__main__":
    # Test the regex extraction module
    test_texts = [
        "Augmentin 1g twice daily for 7 days",
        "Panadol 500mg 1x3",
        "قطرة مرتين يوميا لمدة اسبوعين",
        "Antibiotic Ointment مرهم مساء",
    ]
    
    print("[NLP] Testing regex extraction module:")
    for text in test_texts:
        result = extract_drug_info(text)
        print(f"\nInput: {text}")
        print(f"Drug: {result['drug']}")
        print(f"Form: {result['form']}")
        print(f"Dosage: {result['dosage']}")
        print(f"Frequency: {result['frequency']}")
        print(f"Duration: {result['duration']}")
