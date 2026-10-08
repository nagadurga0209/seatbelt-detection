"""Evaluate a trained YOLOv8 seat-belt detector.

Reports the standard object-detection metrics (precision, recall,
mAP@0.5, mAP@0.5:0.95) on the validation split.

    python -m src.evaluate --weights weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.train import _resolve_data_yaml


def main():
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 model.")
    parser.add_argument(
        "--weights",
        default=str(config.WEIGHTS_DIR / "best.pt"),
        help="Path to trained weights (best.pt).",
    )
    parser.add_argument("--data", default=None, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"Weights not found at {weights}. Train the model first "
            "(python -m src.train) or pass --weights."
        )

    from ultralytics import YOLO

    data_yaml = _resolve_data_yaml(args.data)
    model = YOLO(str(weights))
    metrics = model.val(data=data_yaml, imgsz=args.imgsz, split=args.split)

    box = metrics.box
    print("\n==== Evaluation ====")
    print(f"Precision   : {box.mp:.4f}")
    print(f"Recall      : {box.mr:.4f}")
    print(f"mAP@0.5     : {box.map50:.4f}")
    print(f"mAP@0.5:0.95: {box.map:.4f}")


if __name__ == "__main__":
    main()
