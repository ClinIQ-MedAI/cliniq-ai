import re

def clean_markdown(text: str):
    """Remove markdown formatting from text for JSON compatibility."""
    
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

input_text = "1. Fever or feeling very warm, often with chills 2. Cough, usually dry and persistent 3. Sore throat or scratchy feeling in the throat 4. Runny or stuffy nose 5. Body aches, especially in the back, arms, and legs 6. Headache, sometimes intense 7. Fatigue or feeling unusually weak and tired 8. Sometimes nausea, vomiting, or diarrhea (more common in children) • Symptoms usually start suddenly and can last from a few days up to two weeks. • If you experience difficulty breathing, chest pain, persistent high fever, or worsening symptoms, seek medical attention promptly. Note: This information is general and not a substitute for professional medical advice. Consult a healthcare provider for personalized guidance."

cleaned = clean_markdown(input_text)
print(f"Original length: {len(input_text)}")
print(f"Cleaned length: {len(cleaned)}")
print(f"Contains newlines? {'Yes' if '\\n' in cleaned else 'No'}")
print(f"Count of newlines: {cleaned.count('\\n')}")

# Print first few chars around where split should happen
# "chills 2. Cough" -> "chills\n2. Cough"
match_idx = input_text.find("2. Cough")
print(f"Original around split: {repr(input_text[match_idx-5:match_idx+10])}")

cleaned_match_idx = cleaned.find("2. Cough")
if cleaned_match_idx != -1:
    print(f"Cleaned around split: {repr(cleaned[cleaned_match_idx-5:cleaned_match_idx+10])}")
else:
    print("Could not find '2. Cough' in cleaned text")
