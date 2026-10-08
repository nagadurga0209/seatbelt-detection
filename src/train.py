"""Train a YOLOv8 seat-belt detector.

Trains on the CLAHE-preprocessed dataset by default. Run the preprocessing
step first:

    python -m src.download_data        # fetch dataset -> data/seatbelt
    python -m src.preprocess           # CLAHE -> data/seatbelt_clahe
    python -m src.train                # train YOLOv8

The data.yaml that ships with the Roboflow download (copied into the
processed dataset) is used automatically; override it with --data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src import config


def _resolve_data_yaml(explicit: str | None) -> str:
    if explicit:
        return explicit
    # Prefer the data.yaml inside the processed dataset, then the raw one,
    # then the top-level data/data.yaml template.
    candidates = [
        config.PROCESSED_DATASET_DIR / "data.yaml",
        config.RAW_DATASET_DIR / "data.yaml",
        config.DATA_YAML,
    ]
    for c in candidates:
        if Path(c).exists():
            return str(c)
    raise FileNotFoundError(
        "No data.yaml found. Download/preprocess the dataset first, or pass "
        "--data /path/to/data.yaml"
    )


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 seat-belt detector.")
    parser.add_argument("--data", default=None, help="Path to data.yaml")
    parser.add_argument("--model", default=config.BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch", type=int, default=config.BATCH)
    parser.add_argument("--device", default=config.DEVICE or None)
    parser.add_argument("--name", default="seatbelt_yolov8")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_yaml = _resolve_data_yaml(args.data)
    print(f"[train] data   : {data_yaml}")
    print(f"[train] model  : {args.model}")
    print(f"[train] epochs : {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}")

    model = YOLO(args.model)
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(config.OUTPUTS_DIR),
        name=args.name,
        patience=20,
    )
    print("[train] finished. Best weights saved under:", results.save_dir)
    print(
        "[train] tip: copy the best.pt into weights/ for inference, e.g.\n"
        f"        {results.save_dir}/weights/best.pt"
    )


if __name__ == "__main__":
    main()
