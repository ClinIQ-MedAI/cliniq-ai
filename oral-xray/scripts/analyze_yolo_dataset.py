import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def read_labels(txt_path: Path):
    items = []
    if not txt_path.exists():
        return items
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        items.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return items

def plot_bar(counter: Counter, names, out_path: Path, title: str):
    xs = list(range(len(names)))
    ys = [counter.get(i, 0) for i in xs]

    plt.figure()
    plt.bar(xs, ys)
    plt.xticks(xs, names, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_hist(values, out_path: Path, title: str, bins=60, logy=False):
    plt.figure()
    plt.hist(values, bins=bins)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def draw_samples(images_dir: Path, labels_dir: Path, out_dir: Path, n=16, pad=0.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        print("No images found:", images_dir)
        return
    sample = random.sample(imgs, k=min(n, len(imgs)))

    for p in sample:
        img = Image.open(p).convert("RGB")
        W, H = img.size
        txt = labels_dir / f"{p.stem}.txt"
        anns = read_labels(txt)

        draw = ImageDraw.Draw(img)
        for (cls, xc, yc, w, h) in anns:
            bw = w * W
            bh = h * H
            x1 = (xc * W) - bw / 2
            y1 = (yc * H) - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            if pad > 0:
                px = bw * pad
                py = bh * pad
                x1 -= px; y1 -= py; x2 += px; y2 += py

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            draw.rectangle([x1, y1, x2, y2], width=3)

        img.save(out_dir / p.name, quality=95)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_root", type=str, required=True, help="dataset/processed/yolo_oral")
    ap.add_argument("--out_dir", type=str, default="Oral-Dental/oral-xray/outputs/dataset_report")
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    yolo_root = Path(args.yolo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = yaml.safe_load((yolo_root / "data.yaml").read_text())
    names = data_yaml["names"]
    nc = int(data_yaml["nc"])

    def analyze_split(split: str):
        images_dir = yolo_root / "images" / split
        labels_dir = yolo_root / "labels" / split

        class_counts = Counter()
        areas = []
        aspect = []
        objs_per_img = []

        empty = 0
        for txt in labels_dir.glob("*.txt"):
            anns = read_labels(txt)
            if len(anns) == 0:
                empty += 1
            objs_per_img.append(len(anns))
            for (cls, xc, yc, w, h) in anns:
                if 0 <= cls < nc:
                    class_counts[cls] += 1
                areas.append(w * h)          # normalized area
                if h > 1e-9:
                    aspect.append(w / h)     # normalized aspect ratio

        return {
            "class_counts": class_counts,
            "areas": areas,
            "aspect": aspect,
            "objs_per_img": objs_per_img,
            "empty": empty,
            "num_images": len(list(images_dir.iterdir()))
        }

    train = analyze_split("train")
    val = analyze_split("val")

    # Save summary JSON
    summary = {
        "nc": nc,
        "names": names,
        "train": {
            "num_images": train["num_images"],
            "empty_labels": train["empty"],
            "total_objects": sum(train["class_counts"].values()),
        },
        "val": {
            "num_images": val["num_images"],
            "empty_labels": val["empty"],
            "total_objects": sum(val["class_counts"].values()),
        }
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plots
    plot_bar(train["class_counts"], names, out_dir / "class_counts_train.png", "Class Counts (Train)")
    plot_bar(val["class_counts"], names, out_dir / "class_counts_val.png", "Class Counts (Val)")
    plot_hist(train["areas"], out_dir / "bbox_area_train.png", "BBox Area (Train) - normalized", bins=80, logy=True)
    plot_hist(train["aspect"], out_dir / "bbox_aspect_train.png", "BBox Aspect Ratio (Train)", bins=80, logy=False)
    plot_hist(train["objs_per_img"], out_dir / "objects_per_image_train.png", "Objects per Image (Train)", bins=50, logy=False)

    # Sample visualizations
    draw_samples(
        images_dir=yolo_root / "images" / "train",
        labels_dir=yolo_root / "labels" / "train",
        out_dir=out_dir / "samples_train",
        n=args.samples
    )
    draw_samples(
        images_dir=yolo_root / "images" / "val",
        labels_dir=yolo_root / "labels" / "val",
        out_dir=out_dir / "samples_val",
        n=args.samples
    )

    print("✅ Dataset report saved to:", out_dir)
    print("  - summary.json")
    print("  - class_counts_train.png / class_counts_val.png")
    print("  - bbox_area_train.png / bbox_aspect_train.png / objects_per_image_train.png")
    print("  - samples_train/ , samples_val/")

if __name__ == "__main__":
    main()
