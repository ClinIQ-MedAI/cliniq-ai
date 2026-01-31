# split_data.py
# Split labels.txt into train.txt and val.txt.

import argparse
import os
import random
from config import Paths

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=Paths().data_root)
    p.add_argument("--labels", type=str, default=None)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    paths = Paths(data_root=args.root)
    labels_path = args.labels or paths.labels_file

    with open(labels_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    random.seed(args.seed)
    random.shuffle(lines)

    n_train = int(args.train_ratio * len(lines))
    train_lines = lines[:n_train]
    val_lines = lines[n_train:]

    os.makedirs(os.path.dirname(paths.train_list), exist_ok=True)
    with open(paths.train_list, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(paths.val_list, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print("Done.")
    print("Train:", len(train_lines), "->", paths.train_list)
    print("Val:  ", len(val_lines), "->", paths.val_list)

if __name__ == "__main__":
    main()
