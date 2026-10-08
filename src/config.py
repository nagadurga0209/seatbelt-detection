"""Central configuration for the seat-belt detection project.

Edit the values here (or override them with environment variables / CLI
flags) instead of hard-coding paths and parameters across scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

# Load variables from a local .env file if python-dotenv is installed.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional
    pass

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Project root = the folder that contains this `src/` package.
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
# Raw dataset as downloaded (YOLO format: train/images, train/labels, ...).
RAW_DATASET_DIR = DATA_DIR / "seatbelt"
# CLAHE-preprocessed copy of the dataset (created by preprocess.py).
PROCESSED_DATASET_DIR = DATA_DIR / "seatbelt_clahe"

# data.yaml used for training. Point this at the processed dataset once
# you have run the CLAHE preprocessing step.
DATA_YAML = DATA_DIR / "data.yaml"

OUTPUTS_DIR = ROOT / "runs"          # ultralytics training/val/predict runs
WEIGHTS_DIR = ROOT / "weights"       # exported / best weights

# --------------------------------------------------------------------------
# Model & training
# --------------------------------------------------------------------------
# Pretrained checkpoint to start from. yolov8s.pt matches the write-up
# (best accuracy ~90.6%); yolov8n.pt is the fast/lightweight variant.
BASE_MODEL = os.getenv("SB_BASE_MODEL", "yolov8s.pt")

IMG_SIZE = int(os.getenv("SB_IMG_SIZE", "640"))
EPOCHS = int(os.getenv("SB_EPOCHS", "100"))
BATCH = int(os.getenv("SB_BATCH", "16"))
DEVICE = os.getenv("SB_DEVICE", "")  # "" -> auto, "0" -> GPU 0, "cpu" -> CPU

# --------------------------------------------------------------------------
# CLAHE preprocessing
# --------------------------------------------------------------------------
CLAHE_CLIP_LIMIT = float(os.getenv("SB_CLAHE_CLIP", "2.0"))
CLAHE_TILE_GRID = (8, 8)

# --------------------------------------------------------------------------
# Inference & alerting
# --------------------------------------------------------------------------
CONF_THRESHOLD = float(os.getenv("SB_CONF", "0.35"))

# Class names (as they appear in your dataset's data.yaml) that represent a
# SEAT-BELT VIOLATION. A detection of any of these triggers an alert.
# Adjust these to match your dataset's labels.
VIOLATION_CLASS_NAMES = {
    name.strip().lower()
    for name in os.getenv(
        "SB_VIOLATION_CLASSES",
        "no_seatbelt,no-seatbelt,no seatbelt,without_seatbelt,no,0",
    ).split(",")
    if name.strip()
}

# Minimum seconds between two alerts for the same vehicle (anti-spam).
ALERT_COOLDOWN_SECONDS = int(os.getenv("SB_ALERT_COOLDOWN", "60"))

# --------------------------------------------------------------------------
# Twilio (credentials come from the environment — never hard-code them)
# --------------------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
# Default recipient / vehicle for quick tests (optional).
ALERT_TO_NUMBER = os.getenv("SB_ALERT_TO")
DEFAULT_VEHICLE_NUMBER = os.getenv("SB_VEHICLE", "UNKNOWN")
