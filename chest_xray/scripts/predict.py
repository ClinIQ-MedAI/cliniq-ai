#!/usr/bin/env python3
"""
Inference Script for Chest X-ray Classification

Features:
- Load trained model
- Apply per-class thresholds
- Generate Grad-CAM heatmaps
- Batch inference
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from data.dataset import get_val_transforms
from models.convnext import ConvNeXtClassifier
from utils.gradcam import visualize_predictions, GradCAMPlusPlus


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    ckpt_config = checkpoint.get('config', {})
    num_classes = ckpt_config.get('num_classes', 13)
    class_names = ckpt_config.get('classes', config.data.classes)
    
    # Create model
    model = ConvNeXtClassifier(
        num_classes=num_classes,
        pretrained=False
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get thresholds
    thresholds = checkpoint.get('thresholds', {name: 0.5 for name in class_names})
    
    print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"   Best AUC: {checkpoint.get('best_auc', 'N/A'):.4f}")
    
    return model, class_names, thresholds


def predict_single(
    model,
    image_path: str,
    transform,
    class_names: List[str],
    thresholds: Dict[str, float],
    device: torch.device,
    generate_gradcam: bool = True,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Predict diseases for a single image.
    
    Args:
        model: Trained model
        image_path: Path to chest X-ray image
        transform: Preprocessing transform
        class_names: List of class names
        thresholds: Per-class thresholds
        device: Torch device
        generate_gradcam: Whether to generate Grad-CAM visualization
        output_dir: Directory to save outputs
    
    Returns:
        Prediction results dictionary
    """
    # Load and preprocess image
    original_image = np.array(Image.open(image_path).convert('RGB'))
    
    # Grayscale to 3-channel
    image = Image.open(image_path).convert('L').convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    # Apply thresholds
    predictions = {}
    detected = []
    
    for i, name in enumerate(class_names):
        prob = float(probs[i])
        thresh = thresholds.get(name, 0.5)
        is_positive = prob > thresh
        
        predictions[name] = {
            'probability': prob,
            'threshold': thresh,
            'detected': is_positive
        }
        
        if is_positive:
            detected.append(name)
    
    result = {
        'image_path': image_path,
        'predictions': predictions,
        'detected_diseases': detected,
        'num_diseases': len(detected)
    }
    
    # Generate Grad-CAM
    if generate_gradcam and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        image_name = Path(image_path).stem
        gradcam_path = os.path.join(output_dir, f'{image_name}_gradcam.png')
        
        # Ensure model is in float32 for Grad-CAM to avoid AMP issues
        model.float()
        
        visualize_predictions(
            model=model,
            image_tensor=image_tensor,
            original_image=original_image,
            class_names=class_names,
            probabilities=probs,
            threshold=0.3,
            save_path=gradcam_path
        )
        
        result['gradcam_path'] = gradcam_path
    
    return result


def predict_batch(
    model,
    image_paths: List[str],
    transform,
    class_names: List[str],
    thresholds: Dict[str, float],
    device: torch.device,
    output_dir: Optional[str] = None
) -> List[Dict]:
    """Predict diseases for multiple images."""
    results = []
    
    for image_path in tqdm(image_paths, desc='Predicting'):
        try:
            result = predict_single(
                model=model,
                image_path=image_path,
                transform=transform,
                class_names=class_names,
                thresholds=thresholds,
                device=device,
                generate_gradcam=output_dir is not None,
                output_dir=output_dir
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({'image_path': image_path, 'error': str(e)})
    
    return results


def print_results(result: Dict):
    """Pretty print prediction results."""
    print(f"\n{'='*60}")
    print(f"📷 Image: {Path(result['image_path']).name}")
    print(f"{'='*60}")
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n🔍 Detected Diseases: {result['num_diseases']}")
    
    if result['detected_diseases']:
        print("   " + ", ".join(result['detected_diseases']))
    else:
        print("   No diseases detected (or No Finding)")
    
    print(f"\n📊 All Predictions:")
    print(f"{'Disease':<20} {'Probability':>12} {'Threshold':>10} {'Status':>10}")
    print("-" * 54)
    
    for name, pred in sorted(
        result['predictions'].items(),
        key=lambda x: x[1]['probability'],
        reverse=True
    ):
        prob = pred['probability']
        thresh = pred['threshold']
        status = "✅" if pred['detected'] else "❌"
        print(f"{name:<20} {prob:>11.2%} {thresh:>10.2f} {status:>10}")
    
    if 'gradcam_path' in result:
        print(f"\n🔥 Grad-CAM saved to: {result['gradcam_path']}")


def main():
    parser = argparse.ArgumentParser(description='Chest X-ray Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Directory of images')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Output directory for results')
    parser.add_argument('--no_gradcam', action='store_true',
                        help='Disable Grad-CAM generation')
    parser.add_argument('--save_json', action='store_true',
                        help='Save results as JSON')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model, class_names, thresholds = load_model(args.checkpoint, device)
    
    # Try to get image size from checkpoint config, fallback to current config
    # We load checkpoint again to check config safely
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        ckpt_config = checkpoint.get('config', {})
        input_size = ckpt_config.get('image_size', config.training.image_size)
    except Exception:
        input_size = config.training.image_size
        
    print(f"Using Image Size: {input_size}")
    
    # Transform
    transform = get_val_transforms(input_size)
    
    # Get images
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        exts = ['.png', '.jpg', '.jpeg']
        image_paths = [
            str(p) for p in Path(args.image_dir).iterdir()
            if p.suffix.lower() in exts
        ]
    else:
        print("Error: Provide --image or --image_dir")
        return
    
    print(f"\nProcessing {len(image_paths)} image(s)...")
    
    # Predict
    output_dir = args.output_dir if not args.no_gradcam else None
    results = predict_batch(
        model=model,
        image_paths=image_paths,
        transform=transform,
        class_names=class_names,
        thresholds=thresholds,
        device=device,
        output_dir=output_dir
    )
    
    # Print results
    for result in results:
        print_results(result)
    
    # Save JSON
    if args.save_json:
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, 'predictions.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {json_path}")


if __name__ == '__main__':
    main()
