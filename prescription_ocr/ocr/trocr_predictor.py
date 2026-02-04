"""
TrOCR Predictor Module
Handles loading and inference for the fine-tuned TrOCR model.
"""
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
from typing import Union, List

class TrOCRPredictor:
    """Wrapper for fine-tuned TrOCR model inference."""
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize the TrOCR predictor.
        
        Args:
            model_path: Path to the fine-tuned model checkpoint
            use_gpu: Whether to use GPU acceleration
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"[TrOCR] Loading model from {model_path} (device={self.device})...")
        
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
            print("[TrOCR] ✓ Model loaded successfully")
        except Exception as e:
            print(f"[TrOCR] Error loading model: {e}")
            raise

    def predict(self, images: Union[np.ndarray, Image.Image, List[Union[np.ndarray, Image.Image]]]) ->List[str]:
        """
        Predict text from image crop(s).
        
        Args:
            images: Single image or list of images (numpy array or PIL Image)
            
        Returns:
            List of predicted strings
        """
        if not isinstance(images, list):
            images = [images]
            
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                # Convert numpy array (BGR/RGB) to PIL Image
                if len(img.shape) == 3:
                    # Assume BGR if coming from cv2, convert to RGB
                    img = img[:, :, ::-1] 
                pil_images.append(Image.fromarray(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                raise ValueError("Images must be numpy arrays or PIL Images")

        if not pil_images:
            return []

        # Process inputs
        pixel_values = self.processor(pil_images, return_tensors="pt").pixel_values.to(self.device)

        # Generate output
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        # Decode output
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
