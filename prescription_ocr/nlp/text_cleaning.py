"""
Text Cleaning Module
Normalizes OCR output and applies fuzzy matching for drug name correction.
"""
import re
import json
from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NLP, DRUGS_DB_PATH


def load_drugs_database(db_path: str = None) -> list:
    """
    Load the drugs database for fuzzy matching.
    
    Args:
        db_path: Path to drugs JSON file
        
    Returns:
        List of drug entries
    """
    path = db_path or str(DRUGS_DB_PATH)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("drugs", [])
    except FileNotFoundError:
        print(f"[NLP] Warning: Drugs database not found at {path}")
        return []


def normalize_text(text: str) -> str:
    """
    Basic text normalization.
    
    Steps:
        - Convert to lowercase
        - Remove extra whitespace
        - Normalize common OCR mistakes
        
    Args:
        text: Raw OCR text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Common OCR mistake corrections
    ocr_corrections = {
        '0': 'o',  # Zero to letter O (context-dependent, use carefully)
        '1': 'l',  # One to letter L (context-dependent)
        '|': 'l',  # Pipe to letter L
        '$': 's',  # Dollar to S
    }
    
    # Note: These corrections are aggressive - only apply if needed
    # For now, just return the lowercased, cleaned text
    
    return text


def normalize_dosage(text: str) -> str:
    """
    Normalize dosage patterns in prescription text.
    
    Converts patterns like:
        - "1x3" → "3 times per day"
        - "2x2" → "twice daily (2 tablets)"
        
    Args:
        text: Input text with dosage patterns
        
    Returns:
        Text with normalized dosage expressions
    """
    patterns = NLP["dosage_patterns"]
    
    for pattern, replacement in patterns.items():
        # Case insensitive replacement
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


def remove_special_symbols(text: str) -> str:
    """
    Remove special symbols while preserving Arabic text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Keep: letters (Latin + Arabic), numbers, spaces, basic punctuation
    # Arabic Unicode range: \u0600-\u06FF
    pattern = r'[^\w\s\u0600-\u06FF.,;:\-/]'
    return re.sub(pattern, '', text)


def fuzzy_match_drug(text: str, drugs_db: list = None, threshold: int = None) -> Optional[dict]:
    """
    Find the best matching drug name using fuzzy matching.
    Corrects OCR typos like "Augm3ntin" → "Augmentin"
    
    Args:
        text: Text possibly containing a drug name
        drugs_db: List of drug entries from database
        threshold: Minimum matching score (0-100)
        
    Returns:
        Best matching drug entry or None
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        print("[NLP] Warning: rapidfuzz not installed. Fuzzy matching disabled.")
        return None
    
    if drugs_db is None:
        drugs_db = load_drugs_database()
    
    if not drugs_db:
        return None
    
    threshold = threshold or NLP["fuzzy_threshold"]
    
    # Build list of all drug names (English and Arabic)
    drug_names = []
    name_to_drug = {}
    
    for drug in drugs_db:
        for key in ["name_en", "name_ar"]:
            if key in drug:
                name = drug[key]
                drug_names.append(name)
                name_to_drug[name.lower()] = drug
    
    # Find best match
    text_lower = text.lower().strip()
    
    # Check each word in the text
    words = text_lower.split()
    best_match = None
    best_score = 0
    
    for word in words:
        if len(word) < 3:  # Skip very short words
            continue
            
        result = process.extractOne(
            word, 
            drug_names, 
            scorer=fuzz.ratio,
            score_cutoff=threshold
        )
        
        if result and result[1] > best_score:
            best_match = result[0]
            best_score = result[1]
    
    if best_match:
        return {
            "matched_name": best_match,
            "confidence": best_score,
            "drug_info": name_to_drug.get(best_match.lower())
        }
    
    return None


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline.
    
    Pipeline:
        1. Remove special symbols
        2. Normalize text (lowercase, whitespace)
        3. Normalize dosage patterns
        
    Args:
        text: Raw OCR output
        
    Returns:
        Cleaned and normalized text
    """
    text = remove_special_symbols(text)
    text = normalize_text(text)
    text = normalize_dosage(text)
    return text


def extract_drug_with_fuzzy_match(text: str) -> dict:
    """
    Extract and correct drug names from text using fuzzy matching.
    
    Args:
        text: Input text from OCR
        
    Returns:
        Dictionary with extracted drug information
    """
    drugs_db = load_drugs_database()
    cleaned = clean_text(text)
    
    match_result = fuzzy_match_drug(cleaned, drugs_db)
    
    if match_result:
        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "drug_match": match_result
        }
    
    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "drug_match": None
    }


if __name__ == "__main__":
    # Test the text cleaning module
    test_texts = [
        "Augm3ntin 1g 1x3",
        "panadol 500mg twice daily",
        "بانادول ٥٠٠ مجم",
    ]
    
    print("[NLP] Testing text cleaning module:")
    for text in test_texts:
        result = extract_drug_with_fuzzy_match(text)
        print(f"\nOriginal: {result['original_text']}")
        print(f"Cleaned: {result['cleaned_text']}")
        if result['drug_match']:
            print(f"Drug Match: {result['drug_match']['matched_name']} "
                  f"(confidence: {result['drug_match']['confidence']}%)")
