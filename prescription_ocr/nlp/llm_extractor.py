"""
LLM Extractor Module
Uses LLM to extract structured medication information from prescription OCR text.
Provides better handling of Arabic text and OCR errors than regex-based extraction.
"""

import json
import re
from typing import List, Dict, Optional
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from nlp.llm_client import get_llm_client, is_llm_available
from nlp.regex_extraction import extract_multiple_drugs  # Fallback


# System prompt for prescription extraction
# Optimized System Prompt
EXTRACTION_SYSTEM_PROMPT = """You are an expert Clinical Pharmacist.
Your task is to identify medication names from a NOISY OCR output of a handwritten prescription.

Rules:
1. Extract ALL probable drug names, even if misspelled or repeated.
2. Do NOT require dosage, frequency, or form if they are missing. Mark them as "not specified".
3. Ignore non-medical noise (random letters, names of doctors, etc.).
4. Normalize drug names to their common commercial or generic spelling in English (e.g., 'Voltaren', 'Panadol').
5. Return a valid JSON array of objects.

Output schema:
[
  {
    "drug": "Normalized Name",
    "dosage": "e.g. 500mg or 'not specified'",
    "frequency": "e.g. twice daily or 'not specified'",
    "confidence": "High/Medium/Low based on clarity"
  }
]
"""

def clean_json_response(response: str) -> str:
    """Clean LLM response to extract valid JSON."""
    if not response:
        return "[]"
    
    # Try to find JSON array in response
    match = re.search(r'\[[\s\S]*?\]', response)
    if match:
        return match.group(0)
    
    match = re.search(r'\{[\s\S]*?\}', response)
    if match:
        return f"[{match.group(0)}]"
    
    return "[]"

def filter_candidates(text: str) -> str:
    """
    Pre-process OCR text to reduce noise and highlight drug candidates.
    - Removes short tokens/punctuation
    - Deduplicates words
    - Keeps only potential drug-like tokens
    """
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    
    # Basic cleaning
    tokens = text.split()
    unique_tokens = []
    seen = set()
    
    for token in tokens:
        # Remove punctuation/symbols
        clean_token = re.sub(r'[^\w\s]', '', token)
        
        # Filter: keep if length > 2 and not already seen
        if len(clean_token) > 2 and clean_token.lower() not in seen:
            unique_tokens.append(clean_token)
            seen.add(clean_token.lower())
            
    # Reassemble best candidates
    return " ".join(unique_tokens[:100])  # Limit to first 100 unique candidates to avoid overflow

def extract_medications_llm(ocr_text: str) -> List[Dict]:
    """
    Extract medication information from OCR text using LLM.
    
    Args:
        ocr_text: Raw OCR text from prescription image
        
    Returns:
        List of medication dictionaries
    """
    if not ocr_text or not ocr_text.strip():
        return []
    
    # Check if LLM is available
    if not is_llm_available():
        print("[LLM Extractor] LLM not available, falling back to regex extraction")
        return extract_multiple_drugs(ocr_text)
    
    client = get_llm_client()
    
    # Pre-process / Filter Candidates
    filtered_text = filter_candidates(ocr_text)
    print(f"[LLM Extractor] Filtered input: {filtered_text[:100]}...")
    
    # Build the prompt
    prompt = f"""Analyze these filtered OCR tokens and extract medications:

---
{filtered_text}
---

Return ONLY the JSON array."""
    
    # Get LLM response
    response = client.get_completion(prompt, EXTRACTION_SYSTEM_PROMPT)
    
    if not response:
        print("[LLM Extractor] No response from LLM, falling back to regex")
        return extract_multiple_drugs(ocr_text)
    
    # Parse JSON response
    try:
        json_str = clean_json_response(response)
        medications = json.loads(json_str)
        
        if isinstance(medications, list):
            # Clean up the results
            cleaned = []
            for med in medications:
                if isinstance(med, dict) and med.get("drug"):
                    item = {
                        "drug": med.get("drug"),
                        "drug_corrected": med.get("drug"), # LLM normalizes it
                        "form": med.get("form", "not specified"),
                        "dosage": med.get("dosage", "not specified"),
                        "frequency": med.get("frequency", "not specified"),
                        "match_method": "llm_pharmacist"
                    }
                    cleaned.append(item)
            
            if cleaned:
                print(f"[LLM Extractor] ✓ Extracted {len(cleaned)} medication(s)")
                return cleaned
        
        # No valid results from LLM
        print("[LLM Extractor] No medications found by LLM, trying regex")
        return extract_multiple_drugs(ocr_text)
        
    except json.JSONDecodeError as e:
        print(f"[LLM Extractor] JSON parse error: {e}")
        print(f"[LLM Extractor] Raw response: {response[:200]}")
        return extract_multiple_drugs(ocr_text)


# Convenience function for pipeline integration
def is_llm_extractor_available() -> bool:
    """Check if LLM extractor is available."""
    return is_llm_available()


if __name__ == "__main__":
    # Test the LLM extractor
    test_text = """
    دكتور محمد أحمد
    استشاري طب العيون
    
    قطرة Tobradex مرتين يوميا
    مرهم Voltaren للعين مساءً
    Panadol 500mg 1x3 لمدة اسبوع
    """
    
    print("[Test] Extracting medications from sample text...")
    results = extract_medications_llm(test_text)
    
    print("\n[Results]")
    print(json.dumps(results, indent=2, ensure_ascii=False))
