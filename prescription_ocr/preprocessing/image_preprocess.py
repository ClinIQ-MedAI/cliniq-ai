"""
Image Preprocessing Module
Applies CLAHE, blur, and grayscale conversion for OCR optimization.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PREPROCESSING, PROCESSED_IMAGES_DIR


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the given path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array (BGR format)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances contrast for better OCR results.
    
    Args:
        image: Grayscale image
        
    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(
        clipLimit=PREPROCESSING["clahe_clip_limit"],
        tileGridSize=PREPROCESSING["clahe_tile_grid_size"]
    )
    return clahe.apply(image)


def apply_blur(image: np.ndarray) -> np.ndarray:
    """
    Apply median blur for denoising.
    Reduces noise while preserving edges (important for text).
    
    Args:
        image: Input image
        
    Returns:
        Denoised image
    """
    kernel_size = PREPROCESSING["blur_kernel_size"]
    return cv2.medianBlur(image, kernel_size)


def preprocess(image_path: str, save_output: bool = None) -> np.ndarray:
    """
    Full preprocessing pipeline for prescription images.
    
    Pipeline:
        1. Load image
        2. Convert to grayscale
        3. Apply CLAHE (contrast enhancement)
        4. Apply median blur (denoising)
        
    Args:
        image_path: Path to input image
        save_output: Whether to save the processed image (default from config)
        
    Returns:
        Preprocessed image ready for OCR
    """
    # Use config default if not specified
    if save_output is None:
        save_output = PREPROCESSING["save_processed"]
    
    # Step 1: Load image
    original = load_image(image_path)
    
    # Step 2: Grayscale
    gray = apply_grayscale(original)
    
    # Step 3: CLAHE
    enhanced = apply_clahe(gray)
    
    # Step 4: Blur
    processed = apply_blur(enhanced)
    
    # Optionally save the processed image
    if save_output:
        PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        output_name = Path(image_path).stem + "_processed.png"
        output_path = PROCESSED_IMAGES_DIR / output_name
        cv2.imwrite(str(output_path), processed)
        print(f"[Preprocessing] Saved processed image to: {output_path}")
    
    return processed


def preprocess_for_ocr(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image and return both original and processed versions.
    Useful when you need the original for visualization.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Tuple of (original_image, processed_image)
    """
    original = load_image(image_path)
    processed = preprocess(image_path, save_output=False)
    return original, processed


if __name__ == "__main__":
    # Test the preprocessing module
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        result = preprocess(test_image)
        print(f"[Preprocessing] Output shape: {result.shape}")
    else:
        print("Usage: python image_preprocess.py <image_path>")
