import sys
from pathlib import Path
import yaml

# add oral-xray root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.training.yolo_trainer import train_yolo_with_full_report


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = train_yolo_with_full_report(cfg)
    print("\n✅ DONE")
    print("Run dir:", run_dir)
    print("Presentation assets:", run_dir / "presentation")


if __name__ == "__main__":
    main()
