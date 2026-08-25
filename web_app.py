from __future__ import annotations

import threading
from functools import lru_cache

import cv2
import gradio as gr
import numpy as np
import spaces
import torch
from ultralytics import YOLO


MODEL_NAME = "yolo11n.pt"
DEFAULT_ZONE = [[0.15, 0.78], [0.38, 0.48], [0.72, 0.48], [0.92, 0.78]]
MODEL_LOCK = threading.Lock()


def denormalize_polygon(points: list[list[float]], width: int, height: int) -> list[tuple[int, int]]:
    return [(round(x * width), round(y * height)) for x, y in points]


def point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    """Load YOLO once per Space process."""
    return YOLO(MODEL_NAME)


@spaces.GPU(duration=10)
def analyze_frame(
    frame_rgb: np.ndarray | None,
    confidence: float,
    show_zone: bool,
) -> tuple[np.ndarray | None, str]:
    if frame_rgb is None:
        return None, "### Камера не подключена\nРазрешите браузеру доступ к камере."

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    height, width = frame_bgr.shape[:2]
    polygon = denormalize_polygon(DEFAULT_ZONE, width, height)
    polygon_np = np.array(polygon, dtype=np.int32)

    if show_zone:
        overlay = frame_bgr.copy()
        cv2.fillPoly(overlay, [polygon_np], (0, 190, 255))
        frame_bgr = cv2.addWeighted(overlay, 0.16, frame_bgr, 0.84, 0)
        cv2.polylines(frame_bgr, [polygon_np], True, (0, 190, 255), 2)

    with MODEL_LOCK:
        result = get_model().predict(
            frame_bgr,
            classes=[0],
            conf=float(confidence),
            imgsz=640,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )[0]

    people = 0
    violations = 0
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        foot = ((x1 + x2) // 2, y2)
        in_zone = show_zone and point_in_polygon(foot, polygon)
        people += 1
        violations += int(in_zone)
        color = (0, 0, 235) if in_zone else (40, 190, 80)
        label = "DANGER" if in_zone else "PERSON"
        score = float(box.conf[0])
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame_bgr, foot, 5, color, -1)
        cv2.putText(
            frame_bgr,
            f"{label} {score:.0%}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
        )

    if violations:
        status = f"### 🔴 ТРЕВОГА\nВ опасной зоне: **{violations}** · Всего людей: **{people}**"
    else:
        status = f"### 🟢 Мониторинг\nЛюдей обнаружено: **{people}** · Нарушений: **0**"

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), status


with gr.Blocks(title="Safety Vision") as demo:
    gr.Markdown(
        """
        # Safety Vision
        **Облачный контроль опасной зоны через камеру устройства.**  
        Разрешите доступ к камере — обработка кадров выполняется на сервере.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            camera = gr.Image(
                sources=["webcam"],
                type="numpy",
                streaming=True,
                label="Камера",
            )
            confidence = gr.Slider(
                minimum=0.2,
                maximum=0.9,
                value=0.45,
                step=0.05,
                label="Порог уверенности",
            )
            show_zone = gr.Checkbox(value=True, label="Контролировать опасную зону")
        with gr.Column(scale=1):
            output = gr.Image(label="Результат", streaming=True)
            status = gr.Markdown("### Ожидание камеры")

    camera.stream(
        fn=analyze_frame,
        inputs=[camera, confidence, show_zone],
        outputs=[output, status],
        stream_every=0.5,
        time_limit=600,
        concurrency_limit=1,
        api_name="analyze_frame",
    )
    gr.Markdown(
        """
        > Это демонстрационный MVP, а не сертифицированная система безопасности.  
        > Кадры передаются в облако. Не используйте приложение для конфиденциальных объектов без согласования.
        """
    )


if __name__ == "__main__":
    demo.launch()
