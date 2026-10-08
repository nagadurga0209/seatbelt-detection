# Automated Seat Belt Detection — YOLOv8 + CLAHE

A real-time computer-vision system that detects whether a person in a
vehicle is wearing a seat belt, and raises an **SMS alert** (via Twilio)
when a violation is detected. Built for road-safety / intelligent traffic
monitoring, where manually checking seat-belt compliance across many
vehicles is impractical.

> **Pipeline:** `camera / image → CLAHE preprocessing → YOLOv8 detection →
> seat-belt / no-seat-belt → alert on violation`

---

## Highlights

- **YOLOv8** single-stage object detection (anchor-free, decoupled head) —
  suited to real-time inference. YOLOv8s for best accuracy, YOLOv8n for
  fastest inference.
- **CLAHE preprocessing** (Contrast Limited Adaptive Histogram Equalization)
  to improve detection in **low-light** conditions, where a seat belt has
  low contrast against clothing or a dark interior.
- **Twilio SMS alerting** when a "no seat belt" violation is detected, with
  credentials loaded from the environment (no secrets in code) and a
  cooldown to prevent spam.
- Clean, scripted workflow: download → preprocess → train → evaluate →
  detect. Works locally or on Google Colab.

---

## Project structure

```
SEATBELT_DETECTION/
├── README.md
├── requirements.txt
├── .env.example            # template for secrets (copy to .env)
├── .gitignore
├── data/
│   ├── README.md           # how to get the dataset
│   └── data.yaml           # YOLO dataset config (class names)
├── src/
│   ├── config.py           # central config (paths, params, env vars)
│   ├── download_data.py    # fetch a dataset from Roboflow
│   ├── preprocess.py       # CLAHE (single frame + whole dataset)
│   ├── train.py            # train YOLOv8
│   ├── evaluate.py         # precision / recall / mAP
│   ├── detect.py           # inference on image / video / webcam + alert
│   └── alert.py            # Twilio SMS alert
├── notebooks/
│   ├── seatbelt_yolov8_colab.ipynb        # end-to-end Colab notebook
│   └── legacy_mobilenetv2_classifier.ipynb # earlier baseline (reference)
└── docs/
    └── seatbelt_detection.pdf             # project write-up
```

---

## Quickstart

### 1. Install

```bash
git clone <your-repo-url>
cd SEATBELT_DETECTION
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env      # then edit .env with your keys
```

You need a free [Roboflow API key](https://app.roboflow.com/settings/api)
to download the dataset, and (optionally) Twilio credentials for SMS alerts.

### 3. Get the dataset

```bash
python -m src.download_data            # -> data/seatbelt/
```

See [`data/README.md`](data/README.md) for alternatives and manual setup.
**Check the class names** in the downloaded `data.yaml` and update
`data/data.yaml` + `SB_VIOLATION_CLASSES` to match.

### 4. CLAHE preprocessing

```bash
python -m src.preprocess               # -> data/seatbelt_clahe/
```

### 5. Train

```bash
python -m src.train --model yolov8s.pt --epochs 100 --imgsz 640
# fast/lightweight variant:
python -m src.train --model yolov8n.pt
```

Best weights land in `runs/seatbelt_yolov8/weights/best.pt`. Copy it to
`weights/best.pt` for the inference commands below.

### 6. Evaluate

```bash
python -m src.evaluate --weights weights/best.pt
# -> Precision / Recall / mAP@0.5 / mAP@0.5:0.95
```

### 7. Detect + alert

```bash
# image
python -m src.detect --weights weights/best.pt --source sample.jpg

# video, with SMS alert on a violation
python -m src.detect --weights weights/best.pt --source clip.mp4 \
    --alert --phone +911234567890 --vehicle "AP 39 XX 1234"

# webcam
python -m src.detect --weights weights/best.pt --source 0 --show
```

---

## How it works

**CLAHE** operates on the L (luminance) channel in LAB colour space,
enhancing local contrast in small tiles with a clip limit that avoids
amplifying noise. The same enhancement is applied at training time (to the
whole dataset) and at inference time (per frame), keeping the input
distribution consistent.

**YOLOv8** processes each enhanced frame through a Backbone (feature
extraction, C2f + SPPF), a Neck (multi-scale feature fusion), and a
Detection Head (anchor-free, decoupled classification + box regression).
It outputs bounding boxes, class labels, and confidence scores in a single
forward pass. Training minimises a combined loss: IoU-based box loss +
Distribution Focal Loss (DFL) for localization, plus classification loss.

**Alerting** is a layer on top of detection: when a violation class is
detected above the confidence threshold, `src/alert.py` sends an SMS via the
Twilio API. YOLO handles detection; Twilio handles notification.

---

## Results

Reported in the project write-up (`docs/seatbelt_detection.pdf`); re-run the
training/evaluation above on your dataset to reproduce:

| Model     | Accuracy / notes            | Inference speed |
|-----------|-----------------------------|-----------------|
| YOLOv8s   | ~90.6% (best accuracy)      | moderate        |
| YOLOv5    | ~90% (multi-person)         | moderate        |
| YOLOv8n   | lightweight                 | ~3.5 ms / frame |

> Numbers depend on the dataset and training budget; treat the table as a
> reference, not a guarantee.

---

## Notes & future work

- Increase training-data diversity (lighting, camera angles, vehicle types).
- Export to ONNX / TensorRT for edge deployment.
- Add license-plate recognition to auto-populate the vehicle number in alerts.

## License

Add a license of your choice (e.g. MIT). Dataset licenses belong to their
respective sources on Roboflow Universe.
