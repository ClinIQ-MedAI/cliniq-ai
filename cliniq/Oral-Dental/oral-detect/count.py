from pathlib import Path

DATA_DIR = Path("/home/moabouag/far/oral-detect/oral_yolo_dataset/Data_balanced")

def count_files(path, exts):
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in exts)

def main():
    print(f"Dataset: {DATA_DIR}\n")

    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    LABEL_EXTS = {".txt"}

    # TRAIN
    train_images = count_files(DATA_DIR / "images" / "train", IMG_EXTS)
    train_labels = count_files(DATA_DIR / "labels" / "train", LABEL_EXTS)

    # VAL
    val_images = count_files(DATA_DIR / "images" / "val", IMG_EXTS)
    val_labels = count_files(DATA_DIR / "labels" / "val", LABEL_EXTS)

    print("========== SPLIT COUNTS ==========")
    print(f"Train Images : {train_images}")
    print(f"Train Labels : {train_labels}")
    print(f"Val Images   : {val_images}")
    print(f"Val Labels   : {val_labels}")
    print("===================================")

if __name__ == "__main__":
    main()
