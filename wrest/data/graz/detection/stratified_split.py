import os
import shutil
import random
from collections import Counter
from tqdm import tqdm
import glob

# --- CONFIGURATION ---
BASE_PATH = "/N/scratch/moabouag/grazpedwri"
IMAGES_DIRS = [
    "/N/scratch/moabouag/grazpedwri/images_part1",
    "/N/scratch/moabouag/grazpedwri/images_part2",
    "/N/scratch/moabouag/grazpedwri/images_part3",
    "/N/scratch/moabouag/grazpedwri/images_part4"
]
LABELS_DIR = "/N/scratch/moabouag/grazpedwri/folder_structure/yolov5/labels"
OUTPUT_DIR = "/N/scratch/moabouag/grazpedwri/dataset_v2" # Using v2 to distinguish
TRAIN_RATIO = 0.8
TEXT_CLASS_ID = 8

def parse_and_filter_label(label_path, dst_label_path):
    """
    Reads label file, filters out TEXT_CLASS_ID.
    Writes the cleaned label to dst_label_path.
    Returns: list of remaining class IDs. 
             If list is empty, file is NOT written (but we might return empty list).
    """
    if not os.path.exists(label_path):
        return []
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    valid_lines = []
    class_ids = []
    
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        try:
            cls_id = int(parts[0])
            if cls_id != TEXT_CLASS_ID:
                valid_lines.append(line)
                class_ids.append(cls_id)
        except ValueError:
            pass
            
    # Write only if we have valid classes
    if valid_lines:
        with open(dst_label_path, 'w') as f:
            f.writelines(valid_lines)
            
    return class_ids

def main():
    print("--- Starting Stratified Split (Text Removal) ---")
    
    # Clean output dir
    if os.path.exists(OUTPUT_DIR):
        print(f"Warning: Output directory {OUTPUT_DIR} exists. Merging/Overwriting...")
        # usually safer to not delete blindly, but for this task we assume clean slate or overwrite
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    # 1. Gather Candidates
    all_samples = []
    skipped_text_only = 0
    skipped_no_label = 0
    
    print("Scanning and Filtering...")
    for folder in IMAGES_DIRS:
        files = os.listdir(folder)
        for f in tqdm(files, desc=f"Scanning {os.path.basename(folder)}", leave=False):
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
                img_path = os.path.join(folder, f)
                basename = os.path.splitext(f)[0]
                label_name = basename + ".txt"
                label_path = os.path.join(LABELS_DIR, label_name)
                
                # Check labels BEFORE adding to list
                # We need to read the file to know if it has non-text classes
                if not os.path.exists(label_path):
                    skipped_no_label += 1
                    continue
                
                # We read it temporarily to check content
                # (Optimization: We could delay writing, but we need to know valid classes for stratification)
                with open(label_path, 'r') as lf:
                    lines = lf.readlines()
                
                valid_classes = []
                for line in lines:
                    try:
                        c = int(line.split()[0])
                        if c != TEXT_CLASS_ID:
                            valid_classes.append(c)
                    except:
                        pass
                
                if not valid_classes:
                    # Either file empty or contained only text
                    skipped_text_only += 1
                    continue
                    
                all_samples.append({
                    'img_path': img_path,
                    'label_path': label_path,
                    'classes': valid_classes,
                    'basename': basename,
                    'ext': os.path.splitext(f)[1]
                })

    print(f"Total valid samples: {len(all_samples)}")
    print(f"Skipped (Text-only or Empty): {skipped_text_only}")
    print(f"Skipped (No Label File): {skipped_no_label}")

    # 2. Stratification
    # Count classes across valid samples
    class_counts = Counter()
    for s in all_samples:
        class_counts.update(s['classes'])
        
    print("\nClass Distribution (Post-Filtering):")
    for c, count in class_counts.most_common():
        print(f"  Class {c}: {count}")
        
    # Rarity Sort
    for s in all_samples:
        # Min count among its classes
        s['rarity'] = min(class_counts[c] for c in s['classes'])
        
    all_samples.sort(key=lambda x: x['rarity'])
    
    # 3. Split and Write
    train_count = 0
    val_count = 0
    
    print("\nWriting files...")
    # Interleave split
    for i, s in enumerate(tqdm(all_samples, desc="Splitting")):
        # Probabilistic assignment to keep ratio generally, but deterministic-ish loop
        is_train = random.random() < TRAIN_RATIO
        
        split = 'train' if is_train else 'val'
        if is_train: train_count += 1 
        else: val_count += 1
        
        # Paths
        dst_img = os.path.join(OUTPUT_DIR, 'images', split, s['basename'] + s['ext'])
        dst_label = os.path.join(OUTPUT_DIR, 'labels', split, s['basename'] + ".txt")
        
        # Copy Image
        shutil.copy2(s['img_path'], dst_img)
        
        # Write Filtered Label
        # Re-using the parse filter logic to write
        parse_and_filter_label(s['label_path'], dst_label)

    print(f"\nDone.")
    print(f"Train: {train_count}")
    print(f"Val: {val_count}")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
