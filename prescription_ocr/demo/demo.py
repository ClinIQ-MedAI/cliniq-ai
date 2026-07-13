"""
Demo Application for Prescription OCR
CLI interface with rich output and optional Streamlit integration.
"""
import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.run_pipeline import PrescriptionPipeline
from config import SAMPLES_DIR, PROCESSED_IMAGES_DIR


def print_banner():
    """Print the demo banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🏥 ClinIQ - Prescription OCR Demo                         ║
║   ─────────────────────────────────────────────────         ║
║   Extract structured data from medical prescriptions         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_result(result: dict):
    """Pretty print the extraction results."""
    print("\n" + "─" * 60)
    print("📋 EXTRACTION RESULTS")
    print("─" * 60)
    
    if result.get('status') == 'success':
        print(f"✅ Status: Success")
        print(f"📄 Source: {Path(result['source_image']).name}")
        
        if 'ocr_confidence' in result:
            conf = result['ocr_confidence']
            print(f"🎯 OCR Confidence: {conf['mean']:.1%} (min: {conf['min']:.1%}, max: {conf['max']:.1%})")
        
        print("\n💊 MEDICATIONS FOUND:")
        print("─" * 40)
        
        for i, med in enumerate(result.get('medications', []), 1):
            print(f"\n  [{i}] Medication:")
            
            # Drug name
            drug = med.get('drug_corrected') or med.get('drug') or 'Unknown'
            if med.get('drug_confidence'):
                print(f"      💊 Drug: {drug} (confidence: {med['drug_confidence']}%)")
            else:
                print(f"      💊 Drug: {drug}")
            
            # Category
            if med.get('drug_category'):
                print(f"      📁 Category: {med['drug_category']}")
            
            # Dosage
            if med.get('dosage'):
                print(f"      💉 Dosage: {med['dosage']}")
            
            # Frequency
            if med.get('frequency'):
                print(f"      ⏰ Frequency: {med['frequency']}")
            
            # Duration
            if med.get('duration'):
                print(f"      📅 Duration: {med['duration']}")
            
            # Route
            if med.get('route'):
                print(f"      🛤️  Route: {med['route']}")
        
        if result.get('raw_text'):
            print("\n📝 RAW OCR TEXT:")
            print("─" * 40)
            print(result['raw_text'][:500])
            if len(result['raw_text']) > 500:
                print("... [truncated]")
        
    else:
        print(f"❌ Status: Error")
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "─" * 60)


def run_demo(image_path: str, save_json: bool = True, use_gpu: bool = False):
    """
    Run the demo on a single image.
    
    Args:
        image_path: Path to prescription image
        save_json: Whether to save JSON output
        use_gpu: Use GPU acceleration
    """
    print_banner()
    
    print(f"🔍 Processing: {image_path}")
    print(f"   GPU: {'Enabled' if use_gpu else 'Disabled'}")
    
    # Initialize pipeline
    pipeline = PrescriptionPipeline(use_gpu=use_gpu)
    
    # Determine output path
    output_json = None
    if save_json:
        PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        output_json = str(PROCESSED_IMAGES_DIR / f"{Path(image_path).stem}_result.json")
    
    # Run pipeline
    result = pipeline.run(
        image_path,
        save_visualization=True,
        output_json_path=output_json
    )
    
    # Print results
    print_result(result)
    
    if save_json:
        print(f"\n💾 Results saved to: {output_json}")
    
    # Show where visualization is saved
    vis_path = PROCESSED_IMAGES_DIR / f"{Path(image_path).stem}_ocr_boxes.png"
    if vis_path.exists():
        print(f"🖼️  Visualization saved to: {vis_path}")
    
    return result


def run_batch_demo(input_dir: str, use_gpu: bool = False):
    """
    Run demo on all images in a directory.
    
    Args:
        input_dir: Directory containing prescription images
        use_gpu: Use GPU acceleration
    """
    print_banner()
    
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    images = [f for f in input_path.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"❌ No images found in: {input_dir}")
        return
    
    print(f"📁 Found {len(images)} images in: {input_dir}")
    
    pipeline = PrescriptionPipeline(use_gpu=use_gpu)
    results = pipeline.run_batch(
        [str(img) for img in images],
        output_dir=str(PROCESSED_IMAGES_DIR)
    )
    
    # Summary
    success_count = sum(1 for r in results if r.get('status') == 'success')
    print(f"\n{'='*60}")
    print(f"📊 BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"   Total: {len(results)}")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed: {len(results) - success_count}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ClinIQ Prescription OCR Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --image prescription.png
  python demo.py --image prescription.png --gpu
  python demo.py --batch ./prescriptions/
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to a single prescription image'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Path to directory containing prescription images'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU acceleration for OCR'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save JSON output'
    )
    
    args = parser.parse_args()
    
    if args.image:
        run_demo(args.image, save_json=not args.no_save, use_gpu=args.gpu)
    elif args.batch:
        run_batch_demo(args.batch, use_gpu=args.gpu)
    else:
        # If no arguments, check for sample images
        if SAMPLES_DIR.exists():
            samples = list(SAMPLES_DIR.glob('*.png')) + list(SAMPLES_DIR.glob('*.jpg'))
            if samples:
                print(f"No image specified. Using sample: {samples[0]}")
                run_demo(str(samples[0]), save_json=not args.no_save, use_gpu=args.gpu)
                return
        
        parser.print_help()


if __name__ == "__main__":
    main()
