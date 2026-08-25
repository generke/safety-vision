from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from safety_vision.geometry import normalize_polygon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw and save a Safety Vision danger zone")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0)
    source.add_argument("--source")
    parser.add_argument("--config", default="config.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source: int | str = args.source if args.source is not None else args.camera
    capture = cv2.VideoCapture(source)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read from video source: {source}")

    clean = frame.copy()
    points: list[tuple[int, int]] = []
    window = "Safety Vision - Zone Editor"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = clean.copy()
        if points:
            pts = np.array(points, dtype=np.int32)
            if len(points) >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], (0, 190, 255))
                canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
            cv2.polylines(canvas, [pts], len(points) >= 3, (0, 190, 255), 2)
            for index, point in enumerate(points, start=1):
                cv2.circle(canvas, point, 5, (30, 30, 235), -1)
                cv2.putText(canvas, str(index), (point[0] + 7, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.rectangle(canvas, (10, 10), (620, 45), (20, 20, 20), -1)
        cv2.putText(canvas, "Click: point | Z: undo | R: reset | S/Enter: save | Q/Esc: exit", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("z"), 8) and points:
            points.pop()
        if key == ord("r"):
            points.clear()
        if key in (ord("s"), 13):
            if len(points) < 3:
                print("Add at least three points before saving.")
                continue
            path = Path(args.config)
            config = {}
            if path.exists():
                config = json.loads(path.read_text(encoding="utf-8"))
            height, width = clean.shape[:2]
            config.update({
                "zone_name": config.get("zone_name", "Danger zone"),
                "polygon_normalized": normalize_polygon(points, width, height),
                "confidence": config.get("confidence", 0.45),
                "alarm_cooldown_seconds": config.get("alarm_cooldown_seconds", 2.0),
            })
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved danger zone to {path}")
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

