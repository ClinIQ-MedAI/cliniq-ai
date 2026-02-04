"""
Prescription OCR Pipeline
End-to-end orchestration: Image → Preprocessing → OCR → NLP → JSON
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUT, PROCESSED_IMAGES_DIR
from preprocessing.image_preprocess import preprocess, load_image
from ocr.ocr_engine import OCREngine
from nlp.text_cleaning import clean_text, fuzzy_match_drug, load_drugs_database
from nlp.regex_extraction import extract_drug_info, extract_multiple_drugs

# Try to import LLM extractor (preferred) or semantic matching (fallback)
try:
    from nlp.llm_extractor import extract_medications_llm, is_llm_extractor_available
    LLM_AVAILABLE = is_llm_extractor_available()
    if LLM_AVAILABLE:
        print("[Pipeline] ✓ LLM extractor available")
    else:
        print("[Pipeline] LLM not available, will use regex extraction")
except ImportError:
    LLM_AVAILABLE = False
    print("[Pipeline] LLM extractor not available, using regex extraction")


class PrescriptionPipeline:
    """
    End-to-end pipeline for extracting structured data from prescription images.
    
    Pipeline Flow:
        1. Image Preprocessing (CLAHE + Blur)
        2. OCR (PaddleOCR)
        3. Text Cleaning & Normalization
        4. Information Extraction (Regex + Fuzzy Matching)
        5. Structured JSON Output
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            use_gpu: Whether to use GPU for OCR
        """
        self.ocr_engine = None
        self.use_gpu = use_gpu
        self.drugs_db = load_drugs_database()
        print("[Pipeline] Initialized (lazy OCR loading)")
    
    def _ensure_ocr_engine(self):
        """Lazy initialization of OCR engine."""
        if self.ocr_engine is None:
            from config import OCR
            backend = OCR.get("backend", "easyocr")
            self.ocr_engine = OCREngine(backend=backend, use_gpu=self.use_gpu)
    
    def run(
        self, 
        image_path: str, 
        save_visualization: bool = True,
        output_json_path: Optional[str] = None
    ) -> dict:
        """
        Run the complete pipeline on a prescription image.
        
        Args:
            image_path: Path to the prescription image
            save_visualization: Whether to save OCR visualization
            output_json_path: Optional path to save JSON output
            
        Returns:
            Dictionary with extracted prescription data
        """
        image_path = Path(image_path)
        print(f"\n{'='*60}")
        print(f"[Pipeline] Processing: {image_path.name}")
        print(f"{'='*60}")
        
        # Step 1: Preprocessing
        print("\n[Step 1] Image Preprocessing...")
        original_image = load_image(str(image_path))
        processed_image = preprocess(str(image_path), save_output=True)
        print(f"         → Applied CLAHE + Median Blur")
        
        # Step 2: OCR
        print("\n[Step 2] OCR Text Extraction...")
        self._ensure_ocr_engine()
        ocr_result = self.ocr_engine.extract_text(processed_image)
        print(f"         → Extracted {len(ocr_result['lines'])} text lines")
        
        # Display raw OCR text before NLP processing
        print("\n" + "="*60)
        print("📝 RAW OCR OUTPUT (before NLP)")
        print("="*60)
        
        # Show as combined text
        raw_text = ocr_result['raw_text']
        print("\n📄 Combined Text:")
        print("-"*40)
        print(raw_text)
        
        # Also show line-by-line with confidence for debugging
        print("\n📊 Line-by-Line Details (sorted by confidence):")
        print("-"*40)
        sorted_lines = sorted(ocr_result['lines'], key=lambda x: x['confidence'], reverse=True)
        for i, line in enumerate(sorted_lines, 1):
            conf = line['confidence']
            text = line['text']
            # Mark high-confidence lines
            marker = "✓" if conf >= 0.8 else "?"
            print(f"  {marker} ({conf:.2f}) {text}")
        print("="*60)
        
        # Save visualization
        if save_visualization and ocr_result['boxes']:
            PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            vis_path = PROCESSED_IMAGES_DIR / f"{image_path.stem}_ocr_boxes.png"
            texts = [line['text'] for line in ocr_result['lines']]
            self.ocr_engine.visualize_results(
                original_image, 
                ocr_result['boxes'],
                texts,
                output_path=str(vis_path)
            )
        
        # Step 3: Text Cleaning
        print("\n[Step 3] Text Cleaning & Normalization...")
        raw_text = ocr_result['raw_text']
        cleaned_text = clean_text(raw_text)
        print(f"         → Cleaned text: {cleaned_text[:100]}...")
        
        # Step 4: Information Extraction
        print("\n[Step 4] Information Extraction...")
        
        # Use LLM extraction if available (best for Arabic and OCR errors)
        if LLM_AVAILABLE:
            print("         → Using LLM extraction")
            drugs_info = extract_medications_llm(raw_text)
        else:
            # Fallback to regex-based extraction
            print("         → Using regex extraction (LLM not available)")
            drugs_info = extract_multiple_drugs(raw_text)
            
            # If no drugs found, try single extraction
            if not drugs_info:
                drugs_info = [extract_drug_info(raw_text)]
            
            # Apply fuzzy matching to improve drug name detection
            for drug_info in drugs_info:
                if drug_info.get('drug') or drug_info.get('raw_text'):
                    text_to_match = drug_info.get('drug') or drug_info.get('raw_text', '')
                    fuzzy_result = fuzzy_match_drug(text_to_match, self.drugs_db)
                    if fuzzy_result:
                        drug_info['drug_corrected'] = fuzzy_result['matched_name']
                        drug_info['drug_confidence'] = fuzzy_result['confidence']
                        drug_info['match_method'] = 'fuzzy'
                        drug_info['drug_category'] = fuzzy_result['drug_info'].get('category') if fuzzy_result['drug_info'] else None
        
        print(f"         → Extracted {len(drugs_info)} medication(s)")
        
        # Show extracted medications in detail
        print("\n" + "-"*60)
        print("💊 EXTRACTED MEDICATIONS:")
        print("-"*60)
        for i, med in enumerate(drugs_info, 1):
            print(f"\n  [{i}] Medication:")
            if med.get('category'):
                print(f"      Category: {med['category']}")
            if med.get('drug') or med.get('drug_corrected'):
                drug_name = med.get('drug_corrected') or med.get('drug')
                print(f"      Drug: {drug_name}")
            if med.get('form'):
                print(f"      Form: {med['form']}")
            if med.get('dosage'):
                print(f"      Dosage: {med['dosage']}")
            if med.get('frequency'):
                print(f"      Frequency: {med['frequency']}")
            if med.get('duration'):
                print(f"      Duration: {med['duration']}")
            if med.get('route'):
                print(f"      Route: {med['route']}")
        print("-"*60)
        
        # Step 5: Build Output
        print("\n[Step 5] Building Structured Output...")
        output = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "source_image": str(image_path),
            "medications": drugs_info,
        }
        
        # Include raw text if configured
        if OUTPUT["include_raw_text"]:
            output["raw_text"] = raw_text
            output["cleaned_text"] = cleaned_text
        
        # Include OCR confidence if configured
        if OUTPUT["include_confidence"]:
            confidences = [line['confidence'] for line in ocr_result['lines']]
            output["ocr_confidence"] = {
                "mean": round(sum(confidences) / len(confidences), 3) if confidences else 0,
                "min": min(confidences) if confidences else 0,
                "max": max(confidences) if confidences else 0
            }
        
        # Save JSON output
        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=OUTPUT["json_indent"], ensure_ascii=False)
            print(f"         → Saved JSON to: {output_json_path}")
        
        print(f"\n{'='*60}")
        print("[Pipeline] ✅ Processing Complete!")
        print(f"{'='*60}")
        
        return output
    
    def run_batch(self, image_paths: list, output_dir: str = None) -> list:
        """
        Process multiple prescription images.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory for JSON outputs
            
        Returns:
            List of extraction results
        """
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[Batch] Processing {i}/{len(image_paths)}")
            
            try:
                output_json = None
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_json = Path(output_dir) / f"{Path(image_path).stem}_result.json"
                
                result = self.run(image_path, output_json_path=str(output_json) if output_json else None)
                results.append(result)
                
            except Exception as e:
                print(f"[Batch] Error processing {image_path}: {e}")
                results.append({
                    "status": "error",
                    "source_image": str(image_path),
                    "error": str(e)
                })
        
        return results


def run_pipeline(image_path: str, output_json: str = None) -> dict:
    """
    Convenience function to run the pipeline on a single image.
    
    Args:
        image_path: Path to prescription image
        output_json: Optional JSON output path
        
    Returns:
        Extraction results dictionary
    """
    pipeline = PrescriptionPipeline()
    return pipeline.run(image_path, output_json_path=output_json)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_json = sys.argv[2] if len(sys.argv) > 2 else None
        
        result = run_pipeline(image_path, output_json)
        
        print("\n" + "="*60)
        print("EXTRACTION RESULTS")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Usage: python run_pipeline.py <image_path> [output.json]")
