"""
Synthetic Drug Name Data Generator - v2
Generates realistic prescription-style images for TrOCR fine-tuning.
Includes: drug name only, drug+dosage, drug+abbreviation, drug+frequency
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
except ImportError:
    print("PIL not installed. Run: pip install Pillow")
    raise


class SyntheticDataGenerator:
    """
    Generates synthetic prescription-style images of drug names.
    Creates realistic variations like doctors actually write.
    """
    
    def __init__(
        self,
        drugs_db_path: str,
        fonts_dir: str = None,
        output_dir: str = None,
        english_ratio: float = 0.8,  # 80% English, 20% Arabic
    ):
        self.drugs_db_path = Path(drugs_db_path)
        self.fonts_dir = Path(fonts_dir) if fonts_dir else Path(__file__).parent / "fonts"
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "synthetic_data"
        self.english_ratio = english_ratio
        
        # Load database
        self.db = self._load_database()
        self.drug_names_en = [d["name_en"] for d in self.db.get("drugs", [])]
        self.drug_names_ar = [d["name_ar"] for d in self.db.get("drugs", []) if "name_ar" in d]
        
        # Get dosages, abbreviations, frequencies from DB
        self.dosages = self.db.get("dosages", ["500mg", "250mg", "1g", "100mg"])
        self.abbreviations = list(self.db.get("abbreviations", {}).keys())
        self.frequencies = list(self.db.get("frequencies", {}).keys())
        
        # Load fonts
        self.fonts = self._load_fonts()
        
        # Colors
        self.bg_colors = [
            (255, 255, 255), (250, 248, 240), (245, 245, 245),
            (255, 250, 240), (248, 248, 255), (255, 255, 250),
        ]
        self.text_colors = [
            (0, 0, 0), (0, 0, 139), (25, 25, 112),
            (0, 0, 128), (47, 79, 79), (0, 51, 102),
        ]
        
        print(f"[SyntheticGen] Loaded {len(self.drug_names_en)} English drugs")
        print(f"[SyntheticGen] Loaded {len(self.drug_names_ar)} Arabic drugs")
        print(f"[SyntheticGen] Loaded {len(self.fonts)} fonts")
        print(f"[SyntheticGen] English ratio: {self.english_ratio}")
    
    def _load_database(self) -> dict:
        """Load drugs database with dosages, abbreviations, frequencies."""
        try:
            with open(self.drugs_db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[SyntheticGen] Error loading DB: {e}")
            return {"drugs": [{"name_en": "Panadol"}, {"name_en": "Augmentin"}]}
    
    def _load_fonts(self) -> List[str]:
        """Load handwriting fonts."""
        fonts = []
        
        if self.fonts_dir.exists():
            for ext in ["*.ttf", "*.otf"]:
                fonts.extend([str(f) for f in self.fonts_dir.glob(ext)])
        
        if not fonts:
            for font_dir in ["/usr/share/fonts", Path.home() / ".fonts"]:
                if Path(font_dir).exists():
                    fonts.extend([str(f) for f in Path(font_dir).rglob("*.ttf")][:15])
        
        return fonts if fonts else [None]
    
    def _get_font(self, size: int = 32) -> ImageFont:
        font_path = random.choice(self.fonts)
        try:
            return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        except:
            return ImageFont.load_default()
    
    def _add_noise(self, image: Image.Image, intensity: float = 0.05) -> Image.Image:
        arr = np.array(image)
        noise = np.random.normal(0, intensity * 255, arr.shape).astype(np.int16)
        return Image.fromarray(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    
    def _add_blur(self, image: Image.Image, radius: float = 0.5) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _add_rotation(self, image: Image.Image, max_angle: float = 5) -> Image.Image:
        return image.rotate(random.uniform(-max_angle, max_angle), 
                          fillcolor=self.bg_colors[0], expand=True)
    
    def generate_realistic_variations(self) -> List[str]:
        """
        Generate realistic prescription variations:
        1. Drug name only: Panadol
        2. Drug + dosage: Augmentin 1g
        3. Drug + abbreviation: Panadol tab
        4. Drug + dosage + abbreviation: Brufen 400mg tab
        5. Drug + frequency: Amoxicillin OD
        6. Full prescription: Augmentin 1g tab TDS
        """
        variations = []
        
        # Use english_ratio to determine language mix
        for drug in self.drug_names_en:
            # Type 1: Drug name only (30%)
            variations.append(drug)
            
            # Type 2: Drug + dosage (25%)
            for dos in random.sample(self.dosages, min(3, len(self.dosages))):
                variations.append(f"{drug} {dos}")
            
            # Type 3: Drug + abbreviation (20%)
            for abbr in random.sample(self.abbreviations, min(2, len(self.abbreviations))):
                variations.append(f"{drug} {abbr}")
            
            # Type 4: Drug + dosage + abbreviation (15%)
            for _ in range(2):
                dos = random.choice(self.dosages)
                abbr = random.choice(self.abbreviations)
                variations.append(f"{drug} {dos} {abbr}")
            
            # Type 5: Drug + frequency (5%)
            for freq in random.sample(self.frequencies, min(2, len(self.frequencies))):
                variations.append(f"{drug} {freq}")
            
            # Type 6: Full prescription style (5%)
            dos = random.choice(self.dosages)
            abbr = random.choice(self.abbreviations)
            freq = random.choice(self.frequencies)
            variations.append(f"{drug} {dos} {abbr} {freq}")
        
        # Add some Arabic variations (based on ratio)
        ar_count = int(len(variations) * (1 - self.english_ratio) / self.english_ratio)
        for drug_ar in random.sample(self.drug_names_ar, min(ar_count, len(self.drug_names_ar))):
            variations.append(drug_ar)
            # Arabic with dosage
            dos = random.choice(self.dosages)
            variations.append(f"{drug_ar} {dos}")
        
        random.shuffle(variations)
        return variations
    
    def generate_image(self, text: str, font_size: int = None, augment: bool = True) -> Tuple[Image.Image, str]:
        if font_size is None:
            font_size = random.randint(28, 48)
        
        font = self._get_font(font_size)
        
        # Calculate text size
        dummy = Image.new('RGB', (1, 1))
        bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Create image
        padding = 25
        image = Image.new('RGB', (w + padding*2, h + padding*2), random.choice(self.bg_colors))
        ImageDraw.Draw(image).text((padding - bbox[0], padding - bbox[1]), 
                                    text, font=font, fill=random.choice(self.text_colors))
        
        # Augmentations
        if augment:
            if random.random() < 0.5:
                image = self._add_noise(image, random.uniform(0.02, 0.06))
            if random.random() < 0.3:
                image = self._add_blur(image, random.uniform(0.3, 0.7))
            if random.random() < 0.5:
                image = self._add_rotation(image, random.uniform(1, 4))
        
        return image, text
    
    def generate_dataset(self, num_samples: int = 50000, save: bool = True) -> List[dict]:
        """Generate full training dataset."""
        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        variations = self.generate_realistic_variations()
        samples_per = max(1, num_samples // len(variations))
        
        print(f"[SyntheticGen] {len(variations)} unique variations × {samples_per} samples each")
        print(f"[SyntheticGen] Generating ~{len(variations) * samples_per} samples...")
        
        dataset = []
        idx = 0
        
        for text in variations:
            for _ in range(samples_per):
                try:
                    image, label = self.generate_image(text)
                    
                    if save:
                        filename = f"{idx:06d}.png"
                        path = images_dir / filename
                        image.save(path)
                        dataset.append({"image_path": str(path), "label": label, "filename": filename})
                    else:
                        dataset.append({"image": image, "label": label})
                    
                    idx += 1
                    if idx % 5000 == 0:
                        print(f"[SyntheticGen] Generated {idx} samples...")
                except Exception as e:
                    continue
        
        if save:
            labels_path = self.output_dir / "labels.json"
            with open(labels_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"[SyntheticGen] ✓ Saved {len(dataset)} samples to {self.output_dir}")
        
        return dataset


def download_handwriting_fonts(output_dir: str = None):
    """Download free handwriting fonts from Google Fonts."""
    import urllib.request
    
    output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "fonts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fonts = [
        ("Caveat.ttf", "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat%5Bwght%5D.ttf"),
        ("DancingScript.ttf", "https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf"),
        ("IndieFlower.ttf", "https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf"),
        ("PatrickHand.ttf", "https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf"),
        ("Kalam.ttf", "https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Regular.ttf"),
        ("Handlee.ttf", "https://github.com/google/fonts/raw/main/ofl/handlee/Handlee-Regular.ttf"),
    ]
    
    print(f"[Fonts] Downloading {len(fonts)} handwriting fonts...")
    for name, url in fonts:
        try:
            path = output_dir / name
            if not path.exists():
                print(f"  → {name}")
                urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    print(f"[Fonts] ✓ Done")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic prescription images")
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of samples")
    parser.add_argument("--drugs-db", type=str, default="data/drugs_db.json", help="Drugs DB path")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_training", help="Output dir")
    parser.add_argument("--download-fonts", action="store_true", help="Download fonts first")
    parser.add_argument("--english-ratio", type=float, default=0.8, help="English to Arabic ratio")
    
    args = parser.parse_args()
    
    if args.download_fonts:
        download_handwriting_fonts(args.output_dir + "/fonts")
    
    gen = SyntheticDataGenerator(
        drugs_db_path=args.drugs_db,
        fonts_dir=args.output_dir + "/fonts",
        output_dir=args.output_dir,
        english_ratio=args.english_ratio
    )
    
    dataset = gen.generate_dataset(num_samples=args.num_samples)
    print(f"\n✓ Generated {len(dataset)} training samples")
