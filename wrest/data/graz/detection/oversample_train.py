import os
import shutil
import random
from collections import Counter, defaultdict
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_DIR = "/N/scratch/moabouag/grazpedwri/dataset_v2"
TRAIN_IMGS_DIR = os.path.join(DATASET_DIR, "images", "train")
TRAIN_LBLS_DIR = os.path.join(DATASET_DIR, "labels", "train")

# Classes to balance (Text ID 8 should be gone, but we check anyway)
MIN_INSTANCES = 1500

def get_classes_from_file(label_path):
    entry = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    c = int(line.split()[0])
                    entry.append(c)
                except:
                    pass
    return entry

def main():
    print("--- Starting Oversampling (Train Only) ---")
    
    if not os.path.exists(TRAIN_LBLS_DIR):
        print(f"Error: {TRAIN_LBLS_DIR} not found. Run split script first.")
        return

    # 1. Analyze current train set
    print("Analyzing current distribution...")
    label_files = [f for f in os.listdir(TRAIN_LBLS_DIR) if f.endswith('.txt')]
    
    class_counts = Counter()
    class_to_files = defaultdict(list)
    
    for lf in tqdm(label_files, desc="Scanning labels"):
        path = os.path.join(TRAIN_LBLS_DIR, lf)
        classes = get_classes_from_file(path)
        class_counts.update(classes)
        for c in classes:
            class_to_files[c].append(lf)
            
    print("\nCurrent Train Distribution:")
    sorted_classes = sorted(class_counts.keys())
    for c in sorted_classes:
        print(f"  Class {c}: {class_counts[c]}")
        
    # 2. Oversample
    print(f"\nTargeting minimum {MIN_INSTANCES} instances per class...")
    
    copies_made = 0
    
    for c in sorted_classes:
        count = class_counts[c]
        if count < MIN_INSTANCES:
            deficit = MIN_INSTANCES - count
            candidates = class_to_files[c]
            
            if not candidates:
                print(f"  Warning: Class {c} has 0 instances? Impossible.")
                continue
                
            print(f"  Class {c}: Adding {deficit} copies...")
            
            for i in range(deficit):
                # Pick random file containing this class
                chosen_lbl_name = random.choice(candidates)
                
                # Derive paths
                src_lbl_path = os.path.join(TRAIN_LBLS_DIR, chosen_lbl_name)
                
                # Find corresponding image (extensions vary)
                basename = os.path.splitext(chosen_lbl_name)[0]
                # We need to find the specific image extension. 
                # Optimization: We could store it in step 1, but directory scan is fast enough given 10-20k files.
                # Actually, blindly searching is slow. Let's rely on standard ext check or map from step 1?
                # Let's simple check typical extensions
                found_img_name = None
                for ext in ['.jpg', '.png', '.jpeg', '.bmp', '.tif']:
                    if os.path.exists(os.path.join(TRAIN_IMGS_DIR, basename + ext)):
                        found_img_name = basename + ext
                        break
                
                if not found_img_name:
                    continue # Skip if phantom label
                    
                src_img_path = os.path.join(TRAIN_IMGS_DIR, found_img_name)
                
                # Create names for copy
                # Using a running counter or random hash to avoid collisions if we pick same file twice
                # We use a loop-local unique ID
                copy_suffix = f"_aug{i}" 
                # Note: If we run this script multiple times, we might re-copy copies. 
                # Ideally we filter out filenames with '_aug' from candidates? 
                # For now assuming single run.
                
                new_basename = basename + copy_suffix
                new_img_name = new_basename + os.path.splitext(found_img_name)[1]
                new_lbl_name = new_basename + ".txt"
                
                dst_img_path = os.path.join(TRAIN_IMGS_DIR, new_img_name)
                dst_lbl_path = os.path.join(TRAIN_LBLS_DIR, new_lbl_name)
                
                shutil.copy2(src_img_path, dst_img_path)
                shutil.copy2(src_lbl_path, dst_lbl_path)
                
                copies_made += 1
                
    print(f"\nDone. Created {copies_made} new synthetic samples.")
    print("New train set size: ", len(os.listdir(TRAIN_IMGS_DIR)))

if __name__ == "__main__":
    main()
