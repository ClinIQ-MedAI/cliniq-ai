#!/usr/bin/env python3
import os
import json
import shutil
import argparse
from pathlib import Path

import yaml

def safe_name(name):
    return name.replace(' ', '_').replace('/', '_')

def link_or_copy(src: Path, dst: Path, use_symlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlink:
        try:
            os.symlink(src.resolve(), dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def convert_split(coco_json: Path, images_dir: Path, out_images: Path, out_labels: Path, cat_id_to_idx: dict, use_symlink: bool):
    d = json.load(open(coco_json, "r"))

    # image_id -> info
    imgs = {im["id"]: im for im in d["images"]}

    # init empty labels for all images
    per_image_lines = {img_id: [] for img_id in imgs.keys()}

    missing_count = {}
    for ann in d["annotations"]:
        img_id = ann["image_id"]
        if img_id not in imgs:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue

        W = imgs[img_id]["width"]
        H = imgs[img_id]["height"]
        if W <= 0 or H <= 0:
            continue

        # Skip annotations with unknown category IDs
        cat_id = ann.get("category_id", None)
        if cat_id not in cat_id_to_idx:
            missing_count[cat_id] = missing_count.get(cat_id, 0) + 1
            continue
        
        cls = cat_id_to_idx[cat_id]

        # YOLO normalized
        xc = (x + w / 2.0) / W
        yc = (y + h / 2.0) / H
        ww = w / W
        hh = h / H

        # clamp (just in case)
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        ww = min(max(ww, 0.0), 1.0)
        hh = min(max(hh, 0.0), 1.0)

        per_image_lines[img_id].append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

    if missing_count:
        print("⚠️  Warning: skipped annotations with unknown category_id(s):")
        for mid, c in missing_count.items():
            print(f"   category_id={mid}  count={c}")

    # write images + labels
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for img_id, info in imgs.items():
        src_img = images_dir / info["file_name"]
        dst_img = out_images / info["file_name"]
        link_or_copy(src_img, dst_img, use_symlink=use_symlink)

        # label txt with same stem
        txt = out_labels / f"{Path(info['file_name']).stem}.txt"
        with open(txt, "w") as f:
            f.write("\n".join(per_image_lines[img_id]) + ("\n" if per_image_lines[img_id] else ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", type=str, required=True, help="dataset/raw (has train2017/ val2017/ annotations/)")
    ap.add_argument("--out_root", type=str, required=True, help="dataset/processed/yolo_xxx")
    ap.add_argument("--symlink", action="store_true", help="symlink images instead of copy")
    args = ap.parse_args()
    coco_root = Path(args.coco_root)
    out_root = Path(args.out_root)

    train_json = coco_root / "annotations" / "instances_train2017.json"
    val_json   = coco_root / "annotations" / "instances_val2017.json"

    # Build stable class order from TRAIN categories (sorted by category id)
    train_data = json.load(open(train_json, "r"))
    cats = sorted(train_data["categories"], key=lambda x: x["id"])
    names = [c["name"] for c in cats]
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}  # 0..K-1

    # Save mapping for later use (YOLO + ConvNeXt must match)
    mapping = {
        "names": names,
        "cat_id_to_yolo_idx": cat_id_to_idx,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "classes.json", "w") as f:
        json.dump(mapping, f, indent=2)

    # Convert splits
    convert_split(
        coco_json=train_json,
        images_dir=coco_root / "train2017",
        out_images=out_root / "images" / "train",
        out_labels=out_root / "labels" / "train",
        cat_id_to_idx=cat_id_to_idx,
        use_symlink=args.symlink
    )
    convert_split(
        coco_json=val_json,
        images_dir=coco_root / "val2017",
        out_images=out_root / "images" / "val",
        out_labels=out_root / "labels" / "val",
        cat_id_to_idx=cat_id_to_idx,
        use_symlink=args.symlink
    )

    # Write data.yaml for Ultralytics
    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names
    }
    with open(out_root / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print("✅ Done")
    print("YOLO dataset:", out_root)
    print("data.yaml:", out_root / "data.yaml")
    print("classes.json:", out_root / "classes.json")

if __name__ == "__main__":
    main()
