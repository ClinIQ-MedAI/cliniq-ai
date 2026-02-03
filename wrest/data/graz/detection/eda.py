import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
BASE_DIR = "/N/scratch/moabouag/grazpedwri/dataset_v2"
OUTPUT_DIR = os.path.join(BASE_DIR, "eda_output")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "images", "train")
TRAIN_LBL_DIR = os.path.join(BASE_DIR, "labels", "train")
VAL_IMG_DIR = os.path.join(BASE_DIR, "images", "val")
VAL_LBL_DIR = os.path.join(BASE_DIR, "labels", "val")

# Class ID map (assuming standard grazing ids for display if known, else numeric)
# Based on previous context, we have classes 0-7 remaining after removing text(8)
CLASS_NAMES = {
    0: "wrist", 1: "fingers", 2: "humerus", 3: "radius", 
    4: "ulna", 5: "metacarpal", 6: "distal_phalanx", 7: "proximal_phalanx"
    # Note: These are PLACEHOLDERS. If actual names aren't known, we use ID.
    # The user has NOT provided a class map. I will use "Class X" if not found.
}

def parse_label_file(path):
    """
    Parses a YOLO format label file.
    Returns list of [class_id, x_center, y_center, width, height]
    """
    boxes = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append([int(parts[0])] + [float(x) for x in parts[1:5]])
    return boxes

def analyze_split(name, img_dir, lbl_dir):
    print(f"Analyzing {name} split...")
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    # Filter for image extensions to be safe
    img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif'))]
    
    class_counts = Counter()
    bbox_areas = []
    bbox_aspect_ratios = []
    bbox_counts_per_image = []
    all_centers_x = []
    all_centers_y = []
    
    label_missing = 0
    
    for img_path in tqdm(img_files, desc=f"Scanning {name}"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, basename + ".txt")
        
        if not os.path.exists(lbl_path):
            label_missing += 1
            bbox_counts_per_image.append(0)
            continue
            
        boxes = parse_label_file(lbl_path)
        bbox_counts_per_image.append(len(boxes))
        
        for b in boxes:
            cid, xc, yc, w, h = b
            class_counts[cid] += 1
            bbox_areas.append(w * h)
            if h > 0:
                bbox_aspect_ratios.append(w / h)
            all_centers_x.append(xc)
            all_centers_y.append(yc)
            
    print(f"  Images: {len(img_files)}")
    print(f"  Labels missing: {label_missing}")
    print(f"  Total boxes: {len(bbox_areas)}")
    
    return {
        "class_counts": class_counts,
        "bbox_areas": bbox_areas,
        "bbox_aspect_ratios": bbox_aspect_ratios,
        "bbox_counts_per_image": bbox_counts_per_image,
        "centers_x": all_centers_x,
        "centers_y": all_centers_y,
        "img_files": img_files,
        "lbl_dir": lbl_dir
    }

def plot_class_distributions(train_data, val_data, output_path):
    plt.figure(figsize=(12, 6))
    
    # Merge keys to get all classes
    all_classes = sorted(list(set(train_data['class_counts'].keys()) | set(val_data['class_counts'].keys())))
    
    train_counts = [train_data['class_counts'][c] for c in all_classes]
    val_counts = [val_data['class_counts'][c] for c in all_classes]
    
    x = np.arange(len(all_classes))
    width = 0.35
    
    plt.bar(x - width/2, train_counts, width, label='Train')
    plt.bar(x + width/2, val_counts, width, label='Validation')
    
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    plt.title('Class Distribution: Train vs Val')
    plt.xticks(x, [str(c) for c in all_classes])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(output_path)
    plt.close()

def plot_heatmap(data, title, output_path):
    if not data['centers_x']:
        return
        
    plt.figure(figsize=(8, 8))
    # Invert Y to match image coordinates (0,0 top left) usually visualized as such, 
    # but heatmap follows plot axes (0,0 bottom left). 
    # YOLO coords: 0,0 top-left (usually). But simple scatter:
    
    plt.hist2d(data['centers_x'], data['centers_y'], bins=64, cmap='hot', range=[[0, 1], [0, 1]])
    plt.colorbar(label='Frequency')
    plt.title(f'{title} Object Center Heatmap')
    plt.xlabel('X (Normalized)')
    plt.ylabel('Y (Normalized)')
    # Invert Y axis to match image coordinate system
    plt.gca().invert_yaxis()
    
    plt.savefig(output_path)
    plt.close()

def plot_box_stats(data, title, output_path):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Areas
    sns.histplot(data['bbox_areas'], bins=50, ax=ax[0], kde=True)
    ax[0].set_title(f'{title} BBox Areas (Norm)')
    ax[0].set_xlabel('Area (w*h)')
    
    # Aspect Ratios
    sns.histplot(data['bbox_aspect_ratios'], bins=50, ax=ax[1], kde=True)
    ax[1].set_title(f'{title} Aspect Ratios (w/h)')
    ax[1].set_xlabel('Ratio')
    ax[1].axvline(1.0, color='r', linestyle='--')
    
    # Count per image
    sns.histplot(data['bbox_counts_per_image'], bins=range(min(data['bbox_counts_per_image'] or [0]), max(data['bbox_counts_per_image'] or [0]) + 2), ax=ax[2], kde=False)
    ax[2].set_title(f'{title} Objects per Image')
    ax[2].set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_samples(img_files, lbl_dir, n, title, output_path):
    # Pick n random images
    if len(img_files) < n:
        n = len(img_files)
    
    samples = random.sample(img_files, n)
    
    # Grid size (trying to be square)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n == 1: axes = [axes]
    else: axes = axes.flatten()
    
    for i, img_path in enumerate(samples):
        # Load image
        
        try:
            img = Image.open(img_path)
            
            # Normalize 16-bit images
            if img.mode == 'I' or img.mode == 'I;16':
                arr = np.array(img).astype(float)
                # Linear normalization to 0-255
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-5) * 255
                img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
            else:
                img = img.convert("RGB")
                
            draw = ImageDraw.Draw(img)
            w_img, h_img = img.size
            
            # Load Label
            basename = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, basename + ".txt")
            boxes = parse_label_file(lbl_path)
            
            for b in boxes:
                cid, xc, yc, bw, bh = b
                
                # Convert normalized xywh to pixel xyxy
                x1 = (xc - bw/2) * w_img
                y1 = (yc - bh/2) * h_img
                x2 = (xc + bw/2) * w_img
                y2 = (yc + bh/2) * h_img
                
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Label
                # draw.text((x1, y1), str(cid), fill="red") # simple text
                
            axes[i].imshow(img)
            axes[i].set_title(os.path.basename(img_path))
            axes[i].axis('off')
            
        except Exception as e:
            print(f"Error visualizing {img_path}: {e}")
            
    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    print("--- Starting EDA ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Analyze Train
    train_data = analyze_split("Train", TRAIN_IMG_DIR, TRAIN_LBL_DIR)
    
    # 2. Analyze Val
    val_data = analyze_split("Val", VAL_IMG_DIR, VAL_LBL_DIR)
    
    # 3. Plots
    print("Generating plots...")
    
    # Distribution
    plot_class_distributions(train_data, val_data, os.path.join(OUTPUT_DIR, "class_distribution.png"))
    
    # Train Stats
    plot_heatmap(train_data, "Train", os.path.join(OUTPUT_DIR, "train_heatmap.png"))
    plot_box_stats(train_data, "Train", os.path.join(OUTPUT_DIR, "train_box_stats.png"))
    
    # Val Stats
    plot_heatmap(val_data, "Val", os.path.join(OUTPUT_DIR, "val_heatmap.png"))
    plot_box_stats(val_data, "Val", os.path.join(OUTPUT_DIR, "val_box_stats.png"))
    
    # 4. Visualizations
    print("Generating sample visualizations...")
    visualize_samples(train_data['img_files'], TRAIN_LBL_DIR, 9, "Train Samples", os.path.join(OUTPUT_DIR, "train_samples.png"))
    visualize_samples(val_data['img_files'], VAL_LBL_DIR, 9, "Val Samples", os.path.join(OUTPUT_DIR, "val_samples.png"))
    
    print(f"\nEDA Complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
