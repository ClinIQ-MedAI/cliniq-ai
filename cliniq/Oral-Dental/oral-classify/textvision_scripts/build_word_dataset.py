# build_word_dataset.py
# Convert polygon annotations into word crops + labels.txt
#
# Output format:
#   word_dataset/images/<imgid>_<counter>.jpg
#   word_dataset/labels.txt  =>  filename\ttext\tcounter

import argparse
import os
import json
import shutil
from typing import List
import pandas as pd
from PIL import Image
from tqdm import tqdm
from config import Paths
from utils import polygon_to_bbox, crop_word

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=Paths().data_root)
    p.add_argument("--in_dir", type=str, default=None, help="Raw directory that contains .jpg and .json (default: /content/train)")
    p.add_argument("--out_dir", type=str, default=None, help="Output word dataset directory (default: /content/word_dataset)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    paths = Paths(data_root=args.root)
    in_dir = args.in_dir or paths.train_dir
    out_dir = args.out_dir or paths.word_dataset_dir
    out_img_dir = os.path.join(out_dir, "images")
    labels_file = os.path.join(out_dir, "labels.txt")

    if args.overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_img_dir, exist_ok=True)

    json_files = [f for f in os.listdir(in_dir) if f.endswith(".json")]
    counter = 0

    with open(labels_file, "w", encoding="utf-8") as lf:
        for jf in tqdm(json_files, desc="Cropping words"):
            json_path = os.path.join(in_dir, jf)
            img_name_jpg = jf.replace(".json", ".jpg")
            img_name_png = jf.replace(".json", ".png")

            img_path = os.path.join(in_dir, img_name_jpg)
            if not os.path.exists(img_path):
                img_path = os.path.join(in_dir, img_name_png)
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")

            # Using pandas for compatibility with notebook, but json.load is fine too.
            try:
                data = pd.read_json(json_path)
            except ValueError:
                # Fallback: list of dicts
                with open(json_path, "r", encoding="utf-8") as f:
                    data = pd.DataFrame(json.load(f))

            for _, row in data.iterrows():
                text = str(row.get("text", ""))
                poly = row.get("polygon", None)
                if not isinstance(poly, dict):
                    continue

                bbox = polygon_to_bbox(poly)
                cropped = crop_word(img, bbox)

                # Skip invalid crops
                if cropped.width < 2 or cropped.height < 2:
                    continue

                base = os.path.splitext(os.path.basename(img_path))[0]
                out_name = f"{base}_{counter}.jpg"
                out_path = os.path.join(out_img_dir, out_name)
                cropped.save(out_path)

                lf.write(f"{out_name}\t{text}\t{counter}\n")
                counter += 1

    print("Done.")
    print("Total word crops:", counter)
    print("Images:", out_img_dir)
    print("Labels:", labels_file)

if __name__ == "__main__":
    main()
