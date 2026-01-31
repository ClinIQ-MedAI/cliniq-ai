from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path("/home/moabouag/far/oral-detect/oral_yolo_dataset/Data_balanced")
CLASSES = ['Caries','Ulcer','Tooth Discoloration','Gingivitis']

def count_labels(label_file):
    with open(label_file, "r") as f:
        return [int(line.split()[0]) for line in f]

def main():
    label_paths = list((DATA_DIR / "labels/train").rglob("*.txt")) + \
                  list((DATA_DIR / "labels/val").rglob("*.txt"))

    class_counts = {i: 0 for i in range(len(CLASSES))}

    for lbl in label_paths:
        for cid in count_labels(lbl):
            class_counts[cid] += 1

    print("\n===== CLASS DISTRIBUTION =====")
    for i, name in enumerate(CLASSES):
        print(f"{name:20s} : {class_counts[i]}")

    # Visualization
    plt.figure(figsize=(8,5))
    plt.bar(CLASSES, [class_counts[i] for i in range(len(CLASSES))])
    plt.xticks(rotation=20)
    plt.title("Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Boxes")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=200)
    print("\nSaved: class_distribution.png")

if __name__ == "__main__":
    main()
