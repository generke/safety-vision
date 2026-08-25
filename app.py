from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from safety_vision.geometry import denormalize_polygon, point_in_polygon


COLORS = {"safe": (60, 200, 90), "danger": (30, 30, 235), "zone": (0, 190, 255)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safety Vision real-time danger-zone monitor")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    source.add_argument("--source", help="Video file or RTSP URL")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO person-detection weights")
    parser.add_argument("--ppe-model", help="Optional custom PPE weights")
    parser.add_argument("--no-sound", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    polygon = config.get("polygon_normalized", [])
    if len(polygon) < 3:
        raise ValueError("config.json must contain at least three polygon points")
    if any(len(point) != 2 or not all(0 <= float(v) <= 1 for v in point) for point in polygon):
        raise ValueError("polygon_normalized coordinates must be between 0 and 1")
    return config


def alarm() -> None:
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1200, 180)
    else:
        print("\a", end="", flush=True)


def open_capture(source: int | str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return capture


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    source: int | str = args.source if args.source is not None else args.camera
    capture = open_capture(source)
    person_model = YOLO(args.model)
    ppe_model = YOLO(args.ppe_model) if args.ppe_model else None
    confidence = float(config.get("confidence", 0.45))
    cooldown = float(config.get("alarm_cooldown_seconds", 2.0))
    last_alarm = 0.0
    capture_dir = Path("captures")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            polygon = denormalize_polygon(config["polygon_normalized"], width, height)
            polygon_np = np.array(polygon, dtype=np.int32)

            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon_np], COLORS["zone"])
            frame = cv2.addWeighted(overlay, 0.16, frame, 0.84, 0)
            cv2.polylines(frame, [polygon_np], True, COLORS["zone"], 2)

            results = person_model.predict(frame, classes=[0], conf=confidence, verbose=False)[0]
            people = 0
            violations = 0
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                foot = ((x1 + x2) // 2, y2)
                is_violation = point_in_polygon(foot, polygon)
                people += 1
                violations += int(is_violation)
                color = COLORS["danger"] if is_violation else COLORS["safe"]
                label = "DANGER" if is_violation else "PERSON"
                score = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, foot, 5, color, -1)
                cv2.putText(frame, f"{label} {score:.0%}", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if ppe_model is not None:
                ppe_results = ppe_model.predict(frame, conf=confidence, verbose=False)[0]
                for box in ppe_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    name = str(ppe_results.names[class_id])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 170, 40), 1)
                    cv2.putText(frame, name, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 170, 40), 1)

            if violations and time.monotonic() - last_alarm >= cooldown:
                if not args.no_sound:
                    alarm()
                last_alarm = time.monotonic()

            status_color = COLORS["danger"] if violations else COLORS["safe"]
            status = f"ALERT: {violations}" if violations else "MONITORING"
            cv2.rectangle(frame, (12, 12), (360, 82), (20, 20, 20), -1)
            cv2.putText(frame, status, (26, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"People: {people} | Q exit | S snapshot", (26, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
            cv2.imshow("Safety Vision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                capture_dir.mkdir(exist_ok=True)
                filename = capture_dir / f"safety_vision_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"Saved {filename}")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

