#!/usr/bin/env python3
"""
ClinIQ - Prescription OCR
Main entry point for the prescription text extraction pipeline.

Usage:
    python main.py --image <path_to_image>
    python main.py --image <path_to_image> --output result.json
    python main.py --batch <directory>
    python main.py --demo
"""
import argparse
import sys
from pathlib import Path

# Ensure the package is in the path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="ClinIQ Prescription OCR - Extract structured data from medical prescriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python main.py --image prescription.png
  
  # Process and save JSON output
  python main.py --image prescription.png --output result.json
  
  # Process all images in a directory
  python main.py --batch ./prescriptions/
  
  # Run interactive demo
  python main.py --demo
  
  # Use GPU acceleration
  python main.py --image prescription.png --gpu
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to a prescription image'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save JSON output'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Directory containing prescription images for batch processing'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run the interactive demo'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU acceleration'
    )
    
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Disable OCR bounding box visualization'
    )
    
    args = parser.parse_args()
    
    # Run demo mode
    if args.demo:
        from demo.demo import main as demo_main
        demo_main()
        return
    
    # Process single image
    if args.image:
        from pipeline.run_pipeline import PrescriptionPipeline
        import json
        
        pipeline = PrescriptionPipeline(use_gpu=args.gpu)
        result = pipeline.run(
            args.image,
            save_visualization=not args.no_visualization,
            output_json_path=args.output
        )
        
        # Print summary
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        
        meds = result.get('medications', [])
        for i, med in enumerate(meds, 1):
            drug = med.get('drug_corrected') or med.get('drug') or 'Unknown'
            dosage = med.get('dosage') or 'N/A'
            freq = med.get('frequency') or 'N/A'
            print(f"  [{i}] {drug} - {dosage} - {freq}")
        
        if args.output:
            print(f"\n💾 Full results saved to: {args.output}")
        
        return
    
    # Batch processing
    if args.batch:
        from pipeline.run_pipeline import PrescriptionPipeline
        from config import PROCESSED_IMAGES_DIR
        from utils.helpers import get_image_files
        
        images = get_image_files(args.batch)
        
        if not images:
            print(f"❌ No images found in: {args.batch}")
            sys.exit(1)
        
        print(f"📁 Found {len(images)} images")
        
        pipeline = PrescriptionPipeline(use_gpu=args.gpu)
        results = pipeline.run_batch(
            [str(img) for img in images],
            output_dir=str(PROCESSED_IMAGES_DIR)
        )
        
        # Summary
        success = sum(1 for r in results if r.get('status') == 'success')
        print(f"\n✅ Processed: {success}/{len(results)} images")
        
        return
    
    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    main()
