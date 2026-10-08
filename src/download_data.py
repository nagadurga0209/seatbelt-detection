"""Download a seat-belt detection dataset in YOLOv8 format from Roboflow.

The project does not ship any dataset. This script fetches a public
seat-belt detection dataset from Roboflow Universe into `data/seatbelt/`.

Setup (one-time):
  1. Create a free account at https://roboflow.com and copy your API key
     from https://app.roboflow.com/settings/api
  2. Put it in your environment:  export ROBOFLOW_API_KEY=xxxxxxxx
     (or add it to your .env file)

Usage:
  python -m src.download_data

You can point it at a different public dataset by editing the WORKSPACE /
PROJECT / VERSION constants below, or passing CLI flags. On any Roboflow
Universe dataset page, click "Download Dataset" -> "YOLOv8" -> "show
download code" to get the exact workspace / project / version values.

Suggested public datasets (object detection, 2 classes):
  * bassant/seatbeltfinal
  * seatbeltdet/seatbelt-vehicleoccupancy
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from src import config

# Sensible default — override via flags if it is unavailable or you prefer
# another dataset.
DEFAULT_WORKSPACE = "bassant"
DEFAULT_PROJECT = "seatbeltfinal"
DEFAULT_VERSION = 1


def download(workspace: str, project: str, version: int, api_key: str) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit(
            "The 'roboflow' package is required. Install it with:\n"
            "    pip install roboflow"
        ) from exc

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8")

    # Roboflow downloads into a folder named "<project>-<version>".
    downloaded = Path(dataset.location)
    target = config.RAW_DATASET_DIR
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[download] removing existing {target}")
        shutil.rmtree(target)
    shutil.move(str(downloaded), str(target))
    print(f"[download] dataset ready at {target}")
    print(
        "[download] NOTE: a data.yaml was included inside the dataset. "
        "Check its class names and update data/data.yaml + "
        "SB_VIOLATION_CLASSES in src/config.py to match."
    )
    return target


def main():
    parser = argparse.ArgumentParser(description="Download a Roboflow dataset.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument(
        "--api-key",
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key (defaults to $ROBOFLOW_API_KEY).",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "No Roboflow API key. Set ROBOFLOW_API_KEY or pass --api-key. "
            "Get a free key at https://app.roboflow.com/settings/api"
        )

    download(args.workspace, args.project, args.version, args.api_key)


if __name__ == "__main__":
    main()
