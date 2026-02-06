# inspect_dataset.py
# Visualize a few raw images and overlay polygons from JSON.
#
# Examples:
#   python inspect_dataset.py --split train --n 12
#   python inspect_dataset.py --image eng_AF_012.jpg

import argparse
import os
import json
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Polygon
from config import Paths
from utils import list_images

def show_grid(image_paths, out_path="raw_preview.png", ncols=4):
    n = len(image_paths)
    nrows = (n + ncols - 1)//ncols
    plt.figure(figsize=(4*ncols, 4*nrows))
    for i, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB")
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(img)
        plt.title(os.path.basename(p))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)

def overlay_polygons(img_path, json_path, out_path="overlay.png"):
    img = Image.open(img_path).convert("RGB")
    with open(json_path, "r", encoding="utf-8") as f:
        anns = json.load(f)

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis("off")
    for ann in anns:
        poly = ann["polygon"]
        pts = [(poly["x0"], poly["y0"]),
               (poly["x1"], poly["y1"]),
               (poly["x2"], poly["y2"]),
               (poly["x3"], poly["y3"])]
        plt.gca().add_patch(Polygon(pts, fill=False, linewidth=1))
        plt.text(poly["x0"], poly["y0"]-3, ann.get("text",""), fontsize=7, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    plt.title(os.path.basename(img_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    print("Saved:", out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=Paths().data_root)
    p.add_argument("--split", type=str, default="train", choices=["train","test"])
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--image", type=str, default=None, help="Specific image name like eng_AF_012.jpg")
    p.add_argument("--out", type=str, default="preview.png")
    args = p.parse_args()

    paths = Paths(data_root=args.root)
    data_dir = paths.train_dir if args.split == "train" else paths.test_dir

    if args.image:
        img_path = os.path.join(data_dir, args.image)
        json_path = os.path.join(data_dir, args.image.replace(".jpg",".json"))
        if not os.path.exists(json_path):
            json_path = os.path.join(data_dir, args.image.replace(".png",".json"))
        overlay_polygons(img_path, json_path, out_path=args.out)
    else:
        imgs = list_images(data_dir)
        random.shuffle(imgs)
        show_grid(imgs[:args.n], out_path=args.out)

if __name__ == "__main__":
    main()
