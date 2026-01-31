import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# -----------------------------
# Utils
# -----------------------------
def read_yolo_labels(txt: Path):
    if (not txt.exists()) or txt.stat().st_size == 0:
        return []
    out = []
    for line in txt.read_text().strip().splitlines():
        p = line.strip().split()
        if len(p) != 5:
            continue
        out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out

def plot_bar(counter: Counter, names, out_path: Path, title: str):
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

def plot_hist(values, out_path: Path, title: str, bins=60, logy=False):
    plt.figure()
    plt.hist(values, bins=bins)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close()

def draw_samples(images_dir: Path, labels_dir: Path, out_dir: Path, n=20):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return
    sample = random.sample(imgs, k=min(n, len(imgs)))
    for p in sample:
        img = Image.open(p).convert("RGB")
        W, H = img.size
        txt = labels_dir / f"{p.stem}.txt"
        anns = read_yolo_labels(txt)
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
            anns = read_yolo_labels(txt)
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
            num_images=num_images,
        )

    train = analyze("train")
    val = analyze("val")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "nc": int(nc),
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
        }
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_bar(train["class_counts"], names, out_dir / "class_counts_train.png", "Class Counts (Train)")
    plot_bar(val["class_counts"], names, out_dir / "class_counts_val.png", "Class Counts (Val)")
    plot_hist(train["areas"], out_dir / "bbox_area_train.png", "BBox Area (Train) [normalized]", bins=80, logy=True)
    plot_hist(train["aspect"], out_dir / "bbox_aspect_train.png", "BBox Aspect Ratio (Train)", bins=80, logy=False)
    plot_hist(train["objs_per_img"], out_dir / "objects_per_image_train.png", "Objects per Image (Train)", bins=50, logy=False)

    draw_samples(yolo_root / "images" / "train", yolo_root / "labels" / "train", out_dir / "samples_train", n=samples)
    draw_samples(yolo_root / "images" / "val", yolo_root / "labels" / "val", out_dir / "samples_val", n=samples)

def make_balanced_train_list(yolo_root: Path, alpha=0.5, max_mult=3, seed=42):
    """
    Creates:
      - train_balanced.txt inside yolo_root
      - data_balanced.yaml inside yolo_root
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
        anns = read_yolo_labels(lbl)
        inst = Counter()
        clsset = set()
        for (c, *_rest) in anns:
            if 0 <= c < nc:
                class_counts[c] += 1
                inst[c] += 1
                clsset.add(c)
        img_rel = str(Path(train_rel) / img_path.name)  # relative path for txt file
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

    # build duplicated train list
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

    # write balancing stats for presentation
    stats = {
        "alpha": alpha,
        "max_mult": max_mult,
        "original_counts": {names[i]: int(class_counts.get(i, 0)) for i in range(nc)},
        "balanced_counts_estimate": {names[i]: int(new_counts.get(i, 0)) for i in range(nc)},
        "total_train_entries_after_duplication": len(lines),
    }
    (root / "balancing_stats.json").write_text(json.dumps(stats, indent=2))

    print("✅ Balanced train list:", train_txt)
    print("✅ Balanced data yaml:", out_yaml)
    print("✅ Balancing stats:", root / "balancing_stats.json")
    return out_yaml

def parse_results_csv(run_dir: Path):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None, None
    rows = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return None, None
    return rows, list(rows[0].keys())

def get_col(cols, key):
    if key in cols:
        return key
    for c in cols:
        if key.lower() in c.lower():
            return c
    return None

def save_training_plots(run_dir: Path):
    rows, cols = parse_results_csv(run_dir)
    if rows is None:
        print("⚠️ No results.csv to plot.")
        return None

    pres = run_dir / "presentation"
    pres.mkdir(parents=True, exist_ok=True)

    ep_col = get_col(cols, "epoch")
    epochs = [int(r[ep_col]) for r in rows] if ep_col else list(range(len(rows)))

    def series(col_name):
        c = get_col(cols, col_name)
        if not c:
            return None
        vals = []
        for r in rows:
            try:
                vals.append(float(r[c]))
            except:
                return None
        return vals

    def plot_line(y, fname, title, ylabel):
        if y is None or len(y) != len(epochs):
            return
        plt.figure()
        plt.plot(epochs, y)
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(pres / fname, dpi=240)
        plt.close()

    # losses
    plot_line(series("train/box_loss"), "train_box_loss.png", "Train Box Loss", "loss")
    plot_line(series("train/cls_loss"), "train_cls_loss.png", "Train Cls Loss", "loss")
    plot_line(series("train/dfl_loss"), "train_dfl_loss.png", "Train DFL Loss", "loss")
    plot_line(series("val/box_loss"),   "val_box_loss.png",   "Val Box Loss", "loss")
    plot_line(series("val/cls_loss"),   "val_cls_loss.png",   "Val Cls Loss", "loss")
    plot_line(series("val/dfl_loss"),   "val_dfl_loss.png",   "Val DFL Loss", "loss")

    # metrics
    plot_line(series("metrics/precision(B)"), "precision.png", "Precision (B)", "precision")
    plot_line(series("metrics/recall(B)"),    "recall.png",    "Recall (B)", "recall")
    plot_line(series("metrics/mAP50(B)"),     "map50.png",     "mAP50 (B)", "mAP50")
    plot_line(series("metrics/mAP50-95(B)"),  "map5095.png",   "mAP50-95 (B)", "mAP50-95")

    # best epoch by mAP50-95
    m = series("metrics/mAP50-95(B)")
    best = None
    if m:
        best_i = max(range(len(m)), key=lambda i: m[i])
        best = {"best_epoch": int(epochs[best_i]), "best_map50_95": float(m[best_i])}
        (pres / "best_metric.json").write_text(json.dumps(best, indent=2))

    print("✅ Saved training plots to:", pres)
    return best

def copy_ultralytics_plots(run_dir: Path):
    pres = run_dir / "presentation"
    pres.mkdir(parents=True, exist_ok=True)
    # common plot files created by ultralytics
    for fn in [
        "results.png",
        "confusion_matrix.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]:
        src = run_dir / fn
        if src.exists():
            shutil.copy2(src, pres / fn)

def sample_predict_images(data_yaml_path: Path, k=30):
    data = yaml.safe_load(Path(data_yaml_path).read_text())
    root = Path(data["path"])
    val_rel = data["val"]
    val_dir = root / val_rel
    imgs = [p for p in val_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return []
    return random.sample(imgs, k=min(k, len(imgs)))

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_root", type=str, required=True, help=".../dataset/processed/yolo_oral (contains data.yaml)")
    ap.add_argument("--model", type=str, default="yolov8x.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="Oral-Dental/oral-xray/outputs/experiments")
    ap.add_argument("--name", type=str, default="yolo_oral_run")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)

    # imbalance handling
    ap.add_argument("--balanced", action="store_true", help="use balanced train list")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max_mult", type=int, default=3)

    # presentation
    ap.add_argument("--samples", type=int, default=30, help="how many prediction samples to save")
    args = ap.parse_args()

    random.seed(args.seed)

    yolo_root = Path(args.yolo_root)
    data_yaml = yolo_root / "data.yaml"
    assert data_yaml.exists(), f"Missing {data_yaml}"

    # ---- dataset report (preprocess for presentation, not changing images) ----
    report_dir = Path(args.project) / args.name / "presentation" / "dataset_report"
    make_dataset_report(yolo_root, report_dir, samples=24)

    # ---- optional balancing (creates data_balanced.yaml) ----
    used_data_yaml = data_yaml
    if args.balanced:
        used_data_yaml = make_balanced_train_list(yolo_root, alpha=args.alpha, max_mult=args.max_mult, seed=args.seed)

    # ---- train ----
    model = YOLO(args.model)
    train_res = model.train(
        data=str(used_data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
        patience=args.patience,
        seed=args.seed,

        # anti-overfit / stable
        optimizer="AdamW",
        lr0=0.002,
        weight_decay=0.0005,
        cos_lr=True,
        label_smoothing=0.05,
        save_period=10,

        # medical-friendly augmentation (متزن)
        degrees=5.0,
        translate=0.05,
        scale=0.20,
        shear=0.0,
        perspective=0.0,
        fliplr=0.20,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.10,
        mosaic=0.15,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
    )

    # ulralytics trainer save dir (most reliable)
    run_dir = Path(model.trainer.save_dir)
    print("✅ Run dir:", run_dir)

    # ---- plots + summary ----
    copy_ultralytics_plots(run_dir)
    best = save_training_plots(run_dir)

    # ---- validate best ----
    best_pt = run_dir / "weights" / "best.pt"
    pres = run_dir / "presentation"
    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        best_model.val(
            data=str(used_data_yaml),
            imgsz=args.imgsz,
            device=args.device,
            project=str(pres),
            name="val_best",
            exist_ok=True,
            plots=True
        )
        # predictions samples
        samples = sample_predict_images(used_data_yaml, k=args.samples)
        if samples:
            best_model.predict(
                source=[str(p) for p in samples],
                conf=0.25,
                save=True,
                project=str(pres),
                name="pred_samples",
                exist_ok=True
            )

    # ---- write run summary json ----
    summary = {
        "run_dir": str(run_dir),
        "used_data_yaml": str(used_data_yaml),
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "balanced": bool(args.balanced),
        "alpha": args.alpha,
        "max_mult": args.max_mult,
        "best": best,
    }
    (pres / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print("✅ Presentation folder:", pres)

if __name__ == "__main__":
    main()
