"""
OCR Engine Module - Enhanced Version
Supports multiple OCR backends: EasyOCR (recommended for Arabic) and PaddleOCR.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OCR, VISUALIZATION, PROCESSED_IMAGES_DIR


class OCREngine:
    """
    Multi-backend OCR engine for prescription text extraction.
    Supports EasyOCR (better for Arabic) and PaddleOCR.
    """
    
    def __init__(
        self, 
        backend: str = "easyocr",  # "easyocr" or "paddle"
        languages: List[str] = None, 
        use_gpu: bool = None
    ):
        """
        Initialize the OCR engine.
        
        Args:
            backend: OCR backend to use ("easyocr" or "paddle")
            languages: List of language codes (e.g., ["ar", "en"])
            use_gpu: Whether to use GPU acceleration
        """
        self.backend = backend.lower()
        self.languages = languages or ["ar", "en"]
        self.use_gpu = use_gpu if use_gpu is not None else OCR.get("use_gpu", False)
        self.ocr = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected OCR backend."""
        # Initialize Detection Backend
        if self.backend == "easyocr":
            self._init_easyocr()
        elif self.backend == "paddle":
            self._init_paddleocr()
        else:
            raise ValueError(f"Unknown OCR backend: {self.backend}")

        # Initialize Recognition Backend (TrOCR)
        if OCR.get("use_trocr", False):
            from ocr.trocr_predictor import TrOCRPredictor
            from config import TROCR
            print("[OCR] Initializing TrOCR for recognition refinement...")
            try:
                self.trocr = TrOCRPredictor(
                    model_path=TROCR["model_path"],
                    use_gpu=TROCR["use_gpu"] and self.use_gpu
                )
                print("[OCR] ✓ TrOCR initialized successfully")
            except Exception as e:
                print(f"[OCR] Warning: Failed to initialize TrOCR: {e}")
                self.trocr = None
        else:
            self.trocr = None

    def _init_easyocr(self):
        """Initialize EasyOCR backend."""
        try:
            import easyocr
            print(f"[OCR] Initializing EasyOCR (languages={self.languages}, gpu={self.use_gpu})")
            self.ocr = easyocr.Reader(
                self.languages, 
                gpu=self.use_gpu,
                verbose=False
            )
            print("[OCR] ✓ EasyOCR initialized successfully")
        except ImportError:
            print("[OCR] EasyOCR not installed, falling back to PaddleOCR")
            self.backend = "paddle"
            self._init_paddleocr()
    
    def _init_paddleocr(self):
        """Initialize PaddleOCR backend."""
        from paddleocr import PaddleOCR
        
        # PaddleOCR uses single language code
        lang = "ar" if "ar" in self.languages else "en"
        
        print(f"[OCR] Initializing PaddleOCR (lang={lang}, gpu={self.use_gpu})")
        self.ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=OCR.get("use_angle_cls", True),
            use_gpu=self.use_gpu,
            det_db_thresh=OCR.get("det_db_thresh", 0.3),
            det_db_box_thresh=OCR.get("det_db_box_thresh", 0.5),
            show_log=OCR.get("show_log", False)
        )
        print("[OCR] ✓ PaddleOCR initialized successfully")
    
    def extract_text(self, image: np.ndarray) -> dict:
        """
        Extract text from image using the selected backend.
        
        Args:
            image: Input image (can be grayscale or BGR)
            
        Returns:
            Dictionary with:
                - 'raw_text': Full extracted text as string
                - 'lines': List of detected text lines with boxes and confidence
                - 'boxes': List of bounding box coordinates
        """
        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 1. Detection & Initial Recognition
        if self.backend == "easyocr":
            result = self._extract_easyocr(image)
        else:
            result = self._extract_paddleocr(image)
        
        # 2. Refinement with TrOCR (if enabled)
        if self.trocr and result['lines']:
            print(f"[OCR] Refining {len(result['lines'])} lines with TrOCR...")
            crops = []
            valid_indices = []
            
            for i, line in enumerate(result['lines']):
                box = line['box']
                # Crop logic
                x_coords = [int(p[0]) for p in box]
                y_coords = [int(p[1]) for p in box]
                x_min, x_max = max(0, min(x_coords)), max(x_coords)
                y_min, y_max = max(0, min(y_coords)), max(y_coords)
                
                # Check for valid crop
                if x_max > x_min and y_max > y_min:
                    crop = image[y_min:y_max, x_min:x_max]
                    crops.append(crop)
                    valid_indices.append(i)
            
            if crops:
                # Batch prediction
                refined_texts = self.trocr.predict(crops)
                
                # Update results
                for idx, new_text in zip(valid_indices, refined_texts):
                    old_text = result['lines'][idx]['text']
                    result['lines'][idx]['text'] = new_text
                    # We might want to keep the confidence from detection or set a placeholder
                    # TrOCR generate doesn't give confidence easily, so we keep old or set to 1.0
                    
            # Rebuild raw text
            result['raw_text'] = "\n".join([line['text'] for line in result['lines']])
            
        return result
    
    def _extract_easyocr(self, image: np.ndarray) -> dict:
        """Extract text using EasyOCR."""
        # EasyOCR returns: [[bbox, text, confidence], ...]
        results = self.ocr.readtext(image)
        
        lines = []
        boxes = []
        texts = []
        
        for result in results:
            box = result[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = result[1]
            confidence = result[2]
            
            lines.append({
                "text": text,
                "confidence": round(confidence, 3),
                "box": box
            })
            boxes.append(box)
            texts.append(text)
        
        return {
            "raw_text": "\n".join(texts),
            "lines": lines,
            "boxes": boxes
        }
    
    def _extract_paddleocr(self, image: np.ndarray) -> dict:
        """Extract text using PaddleOCR."""
        result = self.ocr.ocr(image, cls=True)
        
        lines = []
        boxes = []
        texts = []
        
        if result and result[0]:
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                lines.append({
                    "text": text,
                    "confidence": round(confidence, 3),
                    "box": box
                })
                boxes.append(box)
                texts.append(text)
        
        return {
            "raw_text": "\n".join(texts),
            "lines": lines,
            "boxes": boxes
        }
    
    def visualize_results(
        self, 
        image: np.ndarray, 
        boxes: list, 
        texts: list = None,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes on the image for visualization.
        
        Args:
            image: Original image (BGR format)
            boxes: List of bounding boxes from OCR
            texts: Optional list of texts to display
            output_path: Optional path to save the visualization
            
        Returns:
            Image with drawn bounding boxes
        """
        vis_image = image.copy()
        
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        box_color = VISUALIZATION.get("box_color", (0, 255, 0))
        box_thickness = VISUALIZATION.get("box_thickness", 2)
        text_color = VISUALIZATION.get("text_color", (0, 0, 255))
        font_scale = VISUALIZATION.get("font_scale", 0.6)
        
        for i, box in enumerate(boxes):
            pts = np.array(box, dtype=np.int32)
            cv2.polylines(vis_image, [pts], True, box_color, box_thickness)
            
            if texts and i < len(texts):
                x = int(pts[0][0])
                y = int(pts[0][1]) - 5
                cv2.putText(
                    vis_image, 
                    texts[i][:30],
                    (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    text_color, 
                    1
                )
        
        if output_path:
            cv2.imwrite(str(output_path), vis_image)
            print(f"[OCR] Saved visualization to: {output_path}")
        elif VISUALIZATION.get("save_visualization", True):
            PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            default_path = PROCESSED_IMAGES_DIR / "ocr_visualization.png"
            cv2.imwrite(str(default_path), vis_image)
            print(f"[OCR] Saved visualization to: {default_path}")
        
        return vis_image


def extract_text_from_image(image_path: str, backend: str = "easyocr") -> dict:
    """
    Convenience function to extract text from an image file.
    
    Args:
        image_path: Path to image file
        backend: OCR backend ("easyocr" or "paddle")
        
    Returns:
        OCR results dictionary
    """
    engine = OCREngine(backend=backend)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return engine.extract_text(image)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        backend = sys.argv[2] if len(sys.argv) > 2 else "easyocr"
        
        print(f"[OCR] Testing with backend: {backend}")
        result = extract_text_from_image(test_image, backend=backend)
        
        print(f"\n[OCR] Extracted text:\n{result['raw_text']}")
        print(f"\n[OCR] Found {len(result['lines'])} text lines")
        
        # Show confidence scores
        print("\n[OCR] Lines by confidence:")
        sorted_lines = sorted(result['lines'], key=lambda x: x['confidence'], reverse=True)
        for line in sorted_lines:
            conf = line['confidence']
            text = line['text']
            marker = "✓" if conf >= 0.8 else "?"
            print(f"  {marker} ({conf:.2f}) {text}")
    else:
        print("Usage: python ocr_engine.py <image_path> [easyocr|paddle]")
