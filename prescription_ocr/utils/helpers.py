"""
Helper Utilities
Common functions used across the prescription OCR pipeline.
"""
import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save (must be JSON-serializable)
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    print(f"[Utils] Saved JSON to: {path}")


def load_json(path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO format timestamp string
    """
    return datetime.now().isoformat()


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_image_files(directory: str, recursive: bool = False) -> list:
    """
    Get all image files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    dir_path = Path(directory)
    
    if recursive:
        files = dir_path.rglob('*')
    else:
        files = dir_path.iterdir()
    
    return [f for f in files if f.suffix.lower() in image_extensions]


def format_confidence(confidence: float, as_percent: bool = True) -> str:
    """
    Format confidence score for display.
    
    Args:
        confidence: Confidence value (0-1 or 0-100)
        as_percent: Whether to show as percentage
        
    Returns:
        Formatted confidence string
    """
    if confidence <= 1:
        confidence = confidence * 100
    
    if as_percent:
        return f"{confidence:.1f}%"
    return f"{confidence:.3f}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def merge_drug_info(regex_result: dict, fuzzy_result: Optional[dict]) -> dict:
    """
    Merge regex extraction results with fuzzy matching results.
    
    Args:
        regex_result: Results from regex extraction
        fuzzy_result: Results from fuzzy matching (optional)
        
    Returns:
        Merged drug information dictionary
    """
    merged = regex_result.copy()
    
    if fuzzy_result:
        merged['drug_corrected'] = fuzzy_result.get('matched_name')
        merged['drug_confidence'] = fuzzy_result.get('confidence')
        
        if fuzzy_result.get('drug_info'):
            merged['drug_category'] = fuzzy_result['drug_info'].get('category')
            merged['drug_name_ar'] = fuzzy_result['drug_info'].get('name_ar')
    
    return merged
