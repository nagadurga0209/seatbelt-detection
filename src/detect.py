"""Run seat-belt detection on an image, a video, or a webcam stream.

Each frame is CLAHE-enhanced before being passed to the YOLOv8 model (the
same preprocessing used at training time). When a seat-belt *violation*
class is detected above the confidence threshold, an SMS alert is raised
through Twilio (see src/alert.py).

Examples:
    # single image
    python -m src.detect --weights weights/best.pt --source path/to/img.jpg

    # video file, with alerting enabled
    python -m src.detect --weights weights/best.pt --source clip.mp4 \
        --alert --phone +911234567890 --vehicle "AP 39 XX 1234"

    # webcam (device 0)
    python -m src.detect --weights weights/best.pt --source 0 --show
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src import config
from src.alert import send_alert_sms
from src.preprocess import apply_clahe

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _is_violation(label: str) -> bool:
    return label.strip().lower() in config.VIOLATION_CLASS_NAMES


def _annotate_and_check(result, names) -> tuple["cv2.Mat", bool]:
    """Draw boxes on the frame and report whether a violation was found."""
    frame = result.orig_img.copy()
    violation = False
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

        is_viol = _is_violation(label)
        violation = violation or is_viol
        color = (0, 0, 255) if is_viol else (0, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return frame, violation


def run(
    weights: str,
    source: str,
    conf: float,
    alert: bool,
    phone: str | None,
    vehicle: str,
    show: bool,
    save_dir: Path,
):
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names  # {id: label}
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Single image -----------------------------------------------------
    src_path = Path(source)
    if src_path.suffix.lower() in IMAGE_EXTS and src_path.exists():
        image = cv2.imread(str(src_path))
        enhanced = apply_clahe(image)
        result = model.predict(enhanced, conf=conf, verbose=False)[0]
        frame, violation = _annotate_and_check(result, names)
        out_path = save_dir / f"detected_{src_path.name}"
        cv2.imwrite(str(out_path), frame)
        print(f"[detect] saved {out_path} (violation={violation})")
        if violation and alert:
            send_alert_sms(phone, vehicle)
        if show:
            cv2.imshow("seatbelt", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ---- Video file or webcam --------------------------------------------
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    writer = None
    if src_path.suffix.lower() in VIDEO_EXTS:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = save_dir / f"detected_{src_path.name}"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    try:
        while True:
            ok, frame_in = cap.read()
            if not ok:
                break
            enhanced = apply_clahe(frame_in)
            result = model.predict(enhanced, conf=conf, verbose=False)[0]
            frame, violation = _annotate_and_check(result, names)

            if violation and alert:
                send_alert_sms(phone, vehicle)  # cooldown prevents spam

            if writer is not None:
                writer.write(frame)
            if show:
                cv2.imshow("seatbelt", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[detect] saved annotated video to {save_dir}")
        if show:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Seat-belt detection inference.")
    parser.add_argument(
        "--weights", default=str(config.WEIGHTS_DIR / "best.pt")
    )
    parser.add_argument("--source", required=True, help="image/video path or webcam index")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    parser.add_argument("--alert", action="store_true", help="send SMS on violation")
    parser.add_argument("--phone", default=config.ALERT_TO_NUMBER)
    parser.add_argument("--vehicle", default=config.DEFAULT_VEHICLE_NUMBER)
    parser.add_argument("--show", action="store_true", help="display window")
    parser.add_argument("--save-dir", default=str(config.OUTPUTS_DIR / "detect"))
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(
            f"Weights not found at {args.weights}. Train first or pass --weights."
        )

    run(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        alert=args.alert,
        phone=args.phone,
        vehicle=args.vehicle,
        show=args.show,
        save_dir=Path(args.save_dir),
    )


if __name__ == "__main__":
    main()
