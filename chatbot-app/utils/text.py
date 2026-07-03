import re

def clean_markdown(text: str):
    """Remove markdown formatting from text for JSON compatibility."""
    # Remove bold (**text**)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # Remove italic (*text*)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove markdown headers (# text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Handle numbered lists
    text = re.sub(r'\s+(\d+)\.\s+', r'\n\1. ', text)
    
    # Convert markdown lists to plain text with bullet points
    text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
    # Remove markdown table formatting
    text = re.sub(r'\|[-\s|]+\|', '', text)  
    text = re.sub(r'\|\s*', '', text)  
    # Clean up multiple line breaks
    text = re.sub(r'\n\s*\n+', '\n', text)
    # Remove leading/trailing whitespace from lines
    text = '\n'.join(line.strip() for line in text.split('\n'))
    # Clean up any HTML entities
    text = text.replace('‑', '-')
    text = text.replace('°', ' degrees')
    
    return text.strip()
