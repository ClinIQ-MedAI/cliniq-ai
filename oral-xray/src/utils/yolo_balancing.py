import json
import math
import random
from pathlib import Path
from collections import Counter

import yaml

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


def make_balanced_train_list(yolo_root: Path, alpha=0.5, max_mult=3, seed=42):
    """
    Creates inside yolo_root:
      - train_balanced.txt
      - data_balanced.yaml
      - balancing_stats.json
    Returns: Path to data_balanced.yaml
    """
    random.seed(seed)

    data = yaml.safe_load((yolo_root / "data.yaml").read_text())
    names = data["names"]
    nc = int(data["nc"])

    root = Path(data["path"])
    train_rel = data["train"]  # images/train
    images_dir = (root / train_rel).resolve()
    labels_dir = (root / "labels" / "train").resolve()

    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise RuntimeError(f"No train images in {images_dir}")

    class_counts = Counter()
    img_rel_to_clsset = {}
    img_rel_to_inst = {}

    for img_path in images:
        lbl = labels_dir / f"{img_path.stem}.txt"
        anns = _read_yolo_labels(lbl)
        inst = Counter()
        clsset = set()
        for (c, *_rest) in anns:
            if 0 <= c < nc:
                class_counts[c] += 1
                inst[c] += 1
                clsset.add(c)

        img_rel = str(Path(train_rel) / img_path.name)
        img_rel_to_clsset[img_rel] = clsset
        img_rel_to_inst[img_rel] = inst

    max_count = max(class_counts.values()) if class_counts else 1

    # multipliers for rare classes
    class_mult = {}
    for c in range(nc):
        cnt = class_counts.get(c, 1)
        w = (max_count / cnt) ** alpha  # alpha=0.5 => sqrt balancing (safe)
        m = int(math.ceil(w))
        m = max(1, min(max_mult, m))
        class_mult[c] = m

    lines = []
    new_counts = Counter()

    for img_rel, clsset in img_rel_to_clsset.items():
        mult = 1 if not clsset else max(class_mult[c] for c in clsset)
        for _ in range(mult):
            lines.append(img_rel)

        inst = img_rel_to_inst[img_rel]
        for c, k in inst.items():
            new_counts[c] += k * mult

    random.shuffle(lines)

    train_txt = root / "train_balanced.txt"
    train_txt.write_text("\n".join(lines) + "\n")

    data_bal = dict(data)
    data_bal["train"] = "train_balanced.txt"
    out_yaml = root / "data_balanced.yaml"
    out_yaml.write_text(yaml.safe_dump(data_bal, sort_keys=False))

    stats = {
        "alpha": alpha,
        "max_mult": max_mult,
        "original_counts": {names[i]: int(class_counts.get(i, 0)) for i in range(nc)},
        "balanced_counts_estimate": {names[i]: int(new_counts.get(i, 0)) for i in range(nc)},
        "total_train_entries_after_duplication": len(lines),
    }
    (root / "balancing_stats.json").write_text(json.dumps(stats, indent=2))
    return out_yaml
