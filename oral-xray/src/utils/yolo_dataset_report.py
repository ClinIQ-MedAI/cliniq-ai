import json
import random
from pathlib import Path
from collections import Counter

import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _read_yolo_labels(txt: Path):
    if (not txt.exists()) or txt.stat().st_size == 0:
        return []
    out = []
    for line in txt.read_text().strip().splitlines():
        p = line.strip().split()
        if len(p) != 5:
            continue
        out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def _plot_bar(counter: Counter, names, out_path: Path, title: str):
    xs = list(range(len(names)))
    ys = [counter.get(i, 0) for i in xs]
    plt.figure()
    plt.bar(xs, ys)
    plt.xticks(xs, names, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close()


def _plot_hist(values, out_path: Path, title: str, bins=60, logy=False):
    plt.figure()
    plt.hist(values, bins=bins)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close()


def _draw_samples(images_dir: Path, labels_dir: Path, out_dir: Path, n=24):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return
    sample = random.sample(imgs, k=min(n, len(imgs)))

    for p in sample:
        img = Image.open(p).convert("RGB")
        W, H = img.size
        anns = _read_yolo_labels(labels_dir / f"{p.stem}.txt")

        draw = ImageDraw.Draw(img)
        for (cls, xc, yc, w, h) in anns:
            bw = w * W
            bh = h * H
            x1 = (xc * W) - bw / 2
            y1 = (yc * H) - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            draw.rectangle([x1, y1, x2, y2], width=3)

        img.save(out_dir / p.name, quality=95)


def make_dataset_report(yolo_root: Path, out_dir: Path, samples=24):
    """
    Creates:
      - summary.json
      - class_counts_train.png / class_counts_val.png
      - bbox_area_train.png / bbox_aspect_train.png / objects_per_image_train.png
      - samples_train/ , samples_val/
    """
    data = yaml.safe_load((yolo_root / "data.yaml").read_text())
    names = data["names"]
    nc = int(data["nc"])

    def analyze(split: str):
        images_dir = yolo_root / "images" / split
        labels_dir = yolo_root / "labels" / split

        class_counts = Counter()
        areas = []
        aspect = []
        objs_per_img = []
        empty = 0

        for txt in labels_dir.glob("*.txt"):
            anns = _read_yolo_labels(txt)
            if not anns:
                empty += 1
            objs_per_img.append(len(anns))
            for (c, xc, yc, w, h) in anns:
                if 0 <= c < nc:
                    class_counts[c] += 1
                areas.append(w * h)
                if h > 1e-9:
                    aspect.append(w / h)

        num_images = len([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        return dict(
            class_counts=class_counts,
            areas=areas,
            aspect=aspect,
            objs_per_img=objs_per_img,
            empty=empty,
            num_images=num_images
        )

    train = analyze("train")
    val = analyze("val")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "nc": nc,
        "names": names,
        "train": {
            "num_images": train["num_images"],
            "empty_labels": train["empty"],
            "total_objects": int(sum(train["class_counts"].values())),
        },
        "val": {
            "num_images": val["num_images"],
            "empty_labels": val["empty"],
            "total_objects": int(sum(val["class_counts"].values())),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_bar(train["class_counts"], names, out_dir / "class_counts_train.png", "Class Counts (Train)")
    _plot_bar(val["class_counts"], names, out_dir / "class_counts_val.png", "Class Counts (Val)")
    _plot_hist(train["areas"], out_dir / "bbox_area_train.png", "BBox Area (Train) [normalized]", bins=80, logy=True)
    _plot_hist(train["aspect"], out_dir / "bbox_aspect_train.png", "BBox Aspect Ratio (Train)", bins=80, logy=False)
    _plot_hist(train["objs_per_img"], out_dir / "objects_per_image_train.png", "Objects per Image (Train)", bins=50, logy=False)

    _draw_samples(yolo_root / "images" / "train", yolo_root / "labels" / "train", out_dir / "samples_train", n=samples)
    _draw_samples(yolo_root / "images" / "val", yolo_root / "labels" / "val", out_dir / "samples_val", n=samples)
