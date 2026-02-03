from pathlib import Path
from tqdm import tqdm

LABELS_DIR = Path("/N/scratch/moabouag/grazpedwri/dataset_v2/labels/train")

bad_files = 0
total_boxes_before = 0
total_boxes_after = 0

label_files = list(LABELS_DIR.glob("*.txt"))

for lbl in tqdm(label_files, desc="Cleaning labels"):
    with open(lbl, "r") as f:
        lines = f.readlines()

    total_boxes_before += len(lines)

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            c = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except:
            continue

        if not (0 < w <= 1 and 0 < h <= 1):
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1):
            continue

        new_lines.append(f"{c} {x} {y} {w} {h}\n")

    total_boxes_after += len(new_lines)

    if new_lines:
        with open(lbl, "w") as f:
            f.writelines(new_lines)
    else:
        lbl.unlink()
        bad_files += 1

print("\n--- Summary ---")
print(f"Total label files: {len(label_files)}")
print(f"Removed empty label files: {bad_files}")
print(f"Boxes before cleaning: {total_boxes_before}")
print(f"Boxes after cleaning:  {total_boxes_after}")
