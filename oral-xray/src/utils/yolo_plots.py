import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_results_csv(run_dir: Path):
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


def _get_col(cols, key):
    if key in cols:
        return key
    for c in cols:
        if key.lower() in c.lower():
            return c
    return None


def copy_ultralytics_plots(run_dir: Path, pres_dir: Path):
    pres_dir.mkdir(parents=True, exist_ok=True)
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
            shutil.copy2(src, pres_dir / fn)


def save_training_plots(run_dir: Path, pres_dir: Path):
    rows, cols = _parse_results_csv(run_dir)
    if rows is None:
        return None

    pres_dir.mkdir(parents=True, exist_ok=True)
    ep_col = _get_col(cols, "epoch")
    epochs = [int(r[ep_col]) for r in rows] if ep_col else list(range(len(rows)))

    def series(col_name):
        c = _get_col(cols, col_name)
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
        plt.savefig(pres_dir / fname, dpi=240)
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

    m = series("metrics/mAP50-95(B)")
    if m:
        best_i = max(range(len(m)), key=lambda i: m[i])
        best = {"best_epoch": int(epochs[best_i]), "best_map50_95": float(m[best_i])}
        (pres_dir / "best_metric.json").write_text(json.dumps(best, indent=2))
        return best

    return None
