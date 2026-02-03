import argparse
import json
from pathlib import Path
from collections import Counter
from PIL import Image
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_")

def read_yolo_labels(txt: Path):
    if (not txt.exists()) or txt.stat().st_size == 0:
        return []
    out = []
    for line in txt.read_text().strip().splitlines():
        p = line.strip().split()
        if len(p) != 5:
            continue
        cls = int(p[0])
        xc, yc, w, h = map(float, p[1:])
        out.append((cls, xc, yc, w, h))
    return out

def crop_one(im: Image.Image, cls, xc, yc, w, h, pad: float, min_side: int):
    W, H = im.size
    bw = w * W
    bh = h * H
    x1 = xc * W - bw / 2
    y1 = yc * H - bh / 2
    x2 = x1 + bw
    y2 = y1 + bh

    # pad (relative to box size)
    if pad > 0:
        px = bw * pad
        py = bh * pad
        x1 -= px; y1 -= py; x2 += px; y2 += py

    # clip
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(W, int(x2)); y2 = min(H, int(y2))

    if (x2 - x1) < min_side or (y2 - y1) < min_side:
        return None
    return im.crop((x1, y1, x2, y2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_root", required=True, help="dataset/processed/yolo_oral")
    ap.add_argument("--out_root", required=True, help="dataset/processed/crops_oral")
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--min_side", type=int, default=24)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    args = ap.parse_args()

    yolo_root = Path(args.yolo_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data = yaml.safe_load((yolo_root / "data.yaml").read_text())
    names = list(data["names"])
    nc = int(data["nc"])

    # stable folder names: "0_Apical_Periodontitis"
    folders = [f"{i}_{safe_name(names[i])}" for i in range(nc)]

    # save mapping for training/inference
    mapping = {
        "nc": nc,
        "names": names,
        "folders": folders,
        "idx_to_folder": {i: folders[i] for i in range(nc)},
        "folder_to_idx": {folders[i]: i for i in range(nc)},
    }
    (out_root / "classes.json").write_text(json.dumps(mapping, indent=2))

    summary = {"train": {}, "val": {}}

    for split in ["train", "val"]:
        images_dir = yolo_root / "images" / split
        labels_dir = yolo_root / "labels" / split

        # create dirs
        for f in folders:
            (out_root / split / f).mkdir(parents=True, exist_ok=True)

        counts = Counter()
        total_crops = 0

        images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        for img_path in images:
            lbl = labels_dir / f"{img_path.stem}.txt"
            anns = read_yolo_labels(lbl)
            if not anns:
                continue

            im = Image.open(img_path).convert("RGB")

            for k, (cls, xc, yc, w, h) in enumerate(anns):
                if not (0 <= cls < nc):
                    continue
                crop = crop_one(im, cls, xc, yc, w, h, pad=args.pad, min_side=args.min_side)
                if crop is None:
                    continue

                out_dir = out_root / split / folders[cls]
                out_name = f"{img_path.stem}_{k:03d}.jpg"
                crop.save(out_dir / out_name, quality=args.jpeg_quality)

                counts[cls] += 1
                total_crops += 1

        summary[split] = {
            "num_images": len(images),
            "total_crops": total_crops,
            "class_counts": {names[i]: int(counts[i]) for i in range(nc)},
        }
        print(f"✅ {split}: total crops={total_crops}")

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print("✅ Saved:", out_root / "classes.json")
    print("✅ Saved:", out_root / "summary.json")
    print("Crops root:", out_root)

if __name__ == "__main__":
    main()
