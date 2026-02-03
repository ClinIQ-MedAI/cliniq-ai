#!/usr/bin/env python3
"""
Quick inference script for dental X-ray analysis.
Runs YOLO + ConvNeXt pipeline on a single image or directory.

Usage:
    python run_inference.py --image path/to/xray.jpg
    python run_inference.py --dir path/to/images/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import DentalInferencePipeline


# Default model paths (relative to project root)
DEFAULT_YOLO = "outputs/experiments/yolo_v8x_base_1024/weights/best.pt"
DEFAULT_CLASSIFIER = "outputs/classification/convnext_large_20260130_090637/weights/best.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Dental X-ray Inference: YOLO + ConvNeXt Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python run_inference.py --image xray.jpg --visualize
  
  # Directory of images
  python run_inference.py --dir dataset/test/ --output results/
  
  # Custom model paths
  python run_inference.py --image xray.jpg \\
      --yolo path/to/yolo.pt \\
      --classifier path/to/convnext.pt
        """
    )
    
    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Single image path")
    group.add_argument("--dir", type=str, help="Directory of images")
    
    # Models
    parser.add_argument("--yolo", type=str, default=None,
                        help=f"YOLO weights (default: {DEFAULT_YOLO})")
    parser.add_argument("--classifier", type=str, default=None,
                        help=f"Classifier weights (default: {DEFAULT_CLASSIFIER})")
    
    # Output
    parser.add_argument("--output", type=str, default="outputs/inference",
                        help="Output directory (default: outputs/inference)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization images")
    parser.add_argument("--json-only", action="store_true",
                        help="Output JSON only (no visualization)")
    
    # Inference settings
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (default: cuda:0)")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--cls-conf", type=float, default=0.85,
                        help="Classifier confidence for refinement (default: 0.85)")
    parser.add_argument("--no-refine", action="store_true",
                        help="Disable ConvNeXt refinement (YOLO only)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Classification batch size (default: 16)")
    
    args = parser.parse_args()
    
    # Resolve model paths
    yolo_weights = args.yolo or str(PROJECT_ROOT / DEFAULT_YOLO)
    classifier_weights = args.classifier or str(PROJECT_ROOT / DEFAULT_CLASSIFIER)
    
    # Validate paths
    if not Path(yolo_weights).exists():
        print(f"❌ YOLO weights not found: {yolo_weights}")
        print("   Use --yolo to specify the correct path")
        sys.exit(1)
    
    if not args.no_refine and not Path(classifier_weights).exists():
        print(f"❌ Classifier weights not found: {classifier_weights}")
        print("   Use --classifier to specify the correct path, or --no-refine for YOLO-only mode")
        sys.exit(1)
    
    # Initialize pipeline
    print("🔧 Initializing inference pipeline...")
    pipeline = DentalInferencePipeline(
        yolo_weights=yolo_weights,
        classifier_weights=classifier_weights if not args.no_refine else None,
        device=args.device,
        yolo_conf=args.yolo_conf,
        classifier_conf_threshold=args.cls_conf,
        batch_size=args.batch_size,
        enable_refinement=not args.no_refine
    )
    
    # Get images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_dir = Path(args.dir)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
        print(f"📂 Found {len(image_paths)} images in {args.dir}")
    
    if not image_paths:
        print("❌ No images found!")
        sys.exit(1)
    
    # Create output directory (relative to project root)
    if not Path(args.output).is_absolute():
        output_dir = PROJECT_ROOT / args.output
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print(f"\n🔍 Running inference on {len(image_paths)} image(s)...")
    
    all_results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {img_path.name}")
        
        try:
            result = pipeline.predict(
                img_path,
                save_visualization=args.visualize,
                output_dir=str(output_dir)  # Use resolved absolute path
            )
            all_results.append(result)
            
            # Print summary
            print(f"   ✅ {result.num_detections} detections, {result.num_refined} refined")
            print(f"   ⏱️  {result.total_time_ms:.1f}ms total")
            
            # Save individual JSON
            json_path = output_dir / f"{img_path.stem}_result.json"
            json_path.write_text(result.to_json())
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Images processed: {len(all_results)}/{len(image_paths)}")
    total_detections = sum(r.num_detections for r in all_results)
    total_refined = sum(r.num_refined for r in all_results)
    avg_time = sum(r.total_time_ms for r in all_results) / max(len(all_results), 1)
    print(f"Total detections: {total_detections}")
    print(f"Total refined: {total_refined} ({100*total_refined/max(total_detections,1):.1f}%)")
    print(f"Average time: {avg_time:.1f}ms per image")
    print(f"\n📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
