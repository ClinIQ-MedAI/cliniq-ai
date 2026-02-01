import json
import random
from pathlib import Path

import yaml

from ultralytics import YOLO

from src.models.detection import build_yolo
from src.utils.yolo_dataset_report import make_dataset_report
from src.utils.yolo_balancing import make_balanced_train_list
from src.utils.yolo_plots import copy_ultralytics_plots, save_training_plots


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _sample_val_images(data_yaml_path: Path, k=30):
    data = yaml.safe_load(data_yaml_path.read_text())
    root = Path(data["path"])
    val_rel = data["val"]
    val_dir = root / val_rel
    imgs = [p for p in val_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return []
    return random.sample(imgs, k=min(k, len(imgs)))


def train_yolo_with_full_report(cfg: dict):
    """
    cfg: dict loaded from yaml config
    Outputs:
      outputs/experiments/<name>/
        - Ultralytics artifacts
        - presentation/ (dataset_report + plots + val + prediction samples)
    """
    yolo_root = Path(cfg["data"]["yolo_root"])
    data_yaml = yolo_root / "data.yaml"

    project = Path(cfg["experiment"]["project_dir"])
    name = cfg["experiment"]["name"]
    project.mkdir(parents=True, exist_ok=True)

    # 1) Dataset report (presentation)
    pres_root = project / name / "presentation"
    make_dataset_report(yolo_root=yolo_root, out_dir=pres_root / "dataset_report", samples=24)

    # 2) Optional balancing
    used_data_yaml = data_yaml
    if cfg["data"].get("use_balanced", False):
        used_data_yaml = make_balanced_train_list(
            yolo_root=yolo_root,
            alpha=float(cfg["data"].get("alpha", 0.5)),
            max_mult=int(cfg["data"].get("max_mult", 3)),
            seed=int(cfg["train"].get("seed", 42)),
        )

    # 3) Train
    model = build_yolo(cfg["model"]["weights"])

    train_kwargs = dict(
        data=str(used_data_yaml),
        imgsz=int(cfg["train"]["imgsz"]),
        epochs=int(cfg["train"]["epochs"]),
        batch=int(cfg["train"]["batch"]),
        workers=int(cfg["train"]["workers"]),
        device=str(cfg["train"]["device"]),
        project=str(project),
        name=str(name),
        rect=True,
        exist_ok=True,
        plots=True,
        patience=int(cfg["train"].get("patience", 30)),
        seed=int(cfg["train"].get("seed", 42)),

        optimizer=str(cfg["train"].get("optimizer", "AdamW")),
        lr0=float(cfg["train"].get("lr0", 0.002)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0005)),
        cos_lr=bool(cfg["train"].get("cos_lr", True)),
        label_smoothing=float(cfg["train"].get("label_smoothing", 0.05)),
        save_period=int(cfg["train"].get("save_period", 10)),
        cache=cfg["train"].get("cache", False),  # 'disk', 'ram', or False

        # Augment (medical-friendly, conservative)
        degrees=float(cfg["augment"].get("degrees", 5.0)),
        translate=float(cfg["augment"].get("translate", 0.05)),
        scale=float(cfg["augment"].get("scale", 0.20)),
        shear=float(cfg["augment"].get("shear", 0.0)),
        perspective=float(cfg["augment"].get("perspective", 0.0)),
        fliplr=float(cfg["augment"].get("fliplr", 0.20)),
        flipud=float(cfg["augment"].get("flipud", 0.0)),
        hsv_h=float(cfg["augment"].get("hsv_h", 0.0)),
        hsv_s=float(cfg["augment"].get("hsv_s", 0.0)),
        hsv_v=float(cfg["augment"].get("hsv_v", 0.10)),
        mosaic=float(cfg["augment"].get("mosaic", 0.15)),
        close_mosaic=int(cfg["augment"].get("close_mosaic", 10)),
        mixup=float(cfg["augment"].get("mixup", 0.0)),
        copy_paste=float(cfg["augment"].get("copy_paste", 0.0)),
    )

    model.train(**train_kwargs)

    # Run dir produced by ultralytics
    run_dir = Path(model.trainer.save_dir)
    pres_dir = run_dir / "presentation"
    pres_dir.mkdir(parents=True, exist_ok=True)

    # 4) Save plots for presentation
    copy_ultralytics_plots(run_dir, pres_dir)
    best = save_training_plots(run_dir, pres_dir)

    # 5) Validate + sample predictions
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        best_model.val(
            data=str(used_data_yaml),
            imgsz=int(cfg["train"]["imgsz"]),
            device=str(cfg["train"]["device"]),
            project=str(pres_dir),
            name="val_best",
            exist_ok=True,
            plots=True,
        )

        samples = _sample_val_images(used_data_yaml, k=int(cfg["reports"].get("pred_samples", 30)))
        if samples:
            best_model.predict(
                source=[str(p) for p in samples],
                conf=float(cfg["reports"].get("pred_conf", 0.25)),
                save=True,
                project=str(pres_dir),
                name="pred_samples",
                exist_ok=True,
            )

    summary = {
        "run_dir": str(run_dir),
        "used_data_yaml": str(used_data_yaml),
        "best": best,
        "config": cfg,
    }
    (pres_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    return run_dir
