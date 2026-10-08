"""CLAHE preprocessing for the seat-belt dataset.

CLAHE (Contrast Limited Adaptive Histogram Equalization) improves *local*
contrast, which helps reveal a low-contrast seat belt against clothing or a
dark vehicle interior — especially in low-light conditions.

This module provides:
  * apply_clahe(image)         -> enhance a single BGR frame (used at inference)
  * preprocess_dataset(...)    -> build a CLAHE copy of a YOLO-format dataset

Because CLAHE only changes pixel intensities (not geometry), the YOLO label
files are copied across unchanged.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from src import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def apply_clahe(
    image,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    tile_grid_size=config.CLAHE_TILE_GRID,
):
    """Apply CLAHE to the luminance (L) channel of a BGR image.

    Working in LAB space keeps colours intact while enhancing contrast.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _iter_splits(src_root: Path):
    """Yield split names that exist (train / valid / test)."""
    for split in ("train", "valid", "val", "test"):
        if (src_root / split).is_dir():
            yield split


def preprocess_dataset(
    src_root: Path = config.RAW_DATASET_DIR,
    dst_root: Path = config.PROCESSED_DATASET_DIR,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
) -> Path:
    """Create a CLAHE-enhanced copy of a YOLO-format dataset.

    Expected layout under `src_root`:
        <split>/images/*.jpg
        <split>/labels/*.txt
    Labels are copied as-is; images are CLAHE-enhanced.
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if not src_root.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {src_root}. "
            "Download it first (see src/download_data.py or data/README.md)."
        )

    total = 0
    for split in _iter_splits(src_root):
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        out_img_dir = dst_root / split / "images"
        out_lbl_dir = dst_root / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        if img_dir.is_dir():
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"  [skip] unreadable image: {img_path.name}")
                    continue
                enhanced = apply_clahe(image, clip_limit=clip_limit)
                cv2.imwrite(str(out_img_dir / img_path.name), enhanced)
                total += 1

        # Copy label files unchanged.
        if lbl_dir.is_dir():
            for lbl_path in lbl_dir.glob("*.txt"):
                shutil.copy2(lbl_path, out_lbl_dir / lbl_path.name)

        print(f"  [{split}] done")

    # Copy data.yaml if present so the processed dataset is self-contained.
    for yaml_name in ("data.yaml", "data.yml"):
        src_yaml = src_root / yaml_name
        if src_yaml.exists():
            shutil.copy2(src_yaml, dst_root / yaml_name)

    print(f"CLAHE preprocessing complete: {total} images -> {dst_root}")
    return dst_root


def main():
    parser = argparse.ArgumentParser(description="CLAHE-preprocess a YOLO dataset.")
    parser.add_argument("--src", default=str(config.RAW_DATASET_DIR))
    parser.add_argument("--dst", default=str(config.PROCESSED_DATASET_DIR))
    parser.add_argument("--clip", type=float, default=config.CLAHE_CLIP_LIMIT)
    args = parser.parse_args()
    preprocess_dataset(Path(args.src), Path(args.dst), args.clip)


if __name__ == "__main__":
    main()
