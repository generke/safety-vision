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
MODEL_LOCK = threading.Lock()
DEFAULT_CAMERAS = {"Камера 1": []}


def denormalize_polygon(points, width, height):
    return [(round(x * width), round(y * height)) for x, y in points]


def point_in_polygon(point, polygon):
    if len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32), point, False) >= 0


def clone_profiles(profiles):
    source = profiles or DEFAULT_CAMERAS
    return {name: [point[:] for point in points] for name, points in source.items()}


def zone_message(selected, points):
    state = "готова" if len(points) >= 3 else "нужно минимум 3"
    return f"Зона **{selected}**: {len(points)} точек — {state}."


def draw_zone_editor(frame, points):
    if frame is None:
        return None
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    polygon = denormalize_polygon(points, width, height)
    if polygon:
        polygon_np = np.asarray(polygon, dtype=np.int32)
        if len(polygon) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [polygon_np], (255, 180, 0))
            canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
            cv2.polylines(canvas, [polygon_np], True, (255, 135, 0), 3)
        elif len(polygon) >= 2:
            cv2.polylines(canvas, [polygon_np], False, (255, 135, 0), 3)
        for number, point in enumerate(polygon, start=1):
            cv2.circle(canvas, point, 8, (255, 80, 0), -1)
            cv2.putText(canvas, str(number), (point[0] + 9, point[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)
    return canvas


def freeze_frame(frame, selected, profiles):
    if frame is None:
        return None, None, "Сначала включите камеру и дождитесь кадра."
    frozen = frame.copy()
    points = (profiles or DEFAULT_CAMERAS).get(selected, [])
    return frozen, draw_zone_editor(frozen, points), f"Кадр камеры **{selected}** зафиксирован. Ставьте вершины кликами по изображению ниже."


def add_camera(name, profiles):
    profiles = clone_profiles(profiles)
    clean_name = (name or "").strip()
    current = next(iter(profiles))
    if not clean_name:
        return profiles, gr.Dropdown(choices=list(profiles), value=current), "Введите название камеры."
    if clean_name in profiles:
        return profiles, gr.Dropdown(choices=list(profiles), value=clean_name), f"Камера **{clean_name}** уже существует."
    profiles[clean_name] = []
    return profiles, gr.Dropdown(choices=list(profiles), value=clean_name), f"Добавлена **{clean_name}**. Нарисуйте для неё зону."


def delete_camera(selected, profiles):
    profiles = clone_profiles(profiles)
    if len(profiles) == 1:
        return profiles, gr.Dropdown(choices=list(profiles), value=selected), "Нельзя удалить единственную камеру."
    profiles.pop(selected, None)
    new_selected = next(iter(profiles))
    return profiles, gr.Dropdown(choices=list(profiles), value=new_selected), f"Камера **{selected}** удалена."


def clear_zone(selected, profiles, frozen):
    profiles = clone_profiles(profiles)
    profiles[selected] = []
    return profiles, draw_zone_editor(frozen, []), f"Зона камеры **{selected}** очищена. Добавьте минимум 3 точки."


def undo_point(selected, profiles, frozen):
    profiles = clone_profiles(profiles)
    points = profiles.setdefault(selected, [])
    if points:
        points.pop()
    return profiles, draw_zone_editor(frozen, points), zone_message(selected, points)


def add_zone_point(selected, profiles, frozen, evt: gr.SelectData):
    profiles = clone_profiles(profiles)
    if frozen is None or not isinstance(evt.index, (list, tuple)) or len(evt.index) < 2:
        return profiles, None, "Сначала нажмите «Зафиксировать кадр для разметки»."
    height, width = frozen.shape[:2]
    x, y = evt.index[:2]
    profiles.setdefault(selected, []).append([
        min(1.0, max(0.0, float(x) / width)),
        min(1.0, max(0.0, float(y) / height)),
    ])
    return profiles, draw_zone_editor(frozen, profiles[selected]), zone_message(selected, profiles[selected])


def save_zone(selected, profiles):
    points = (profiles or {}).get(selected, [])
    if len(points) < 3:
        return f"❌ Зона **{selected}** не сохранена: поставьте минимум 3 точки."
    return f"✅ Зона **{selected}** сохранена: {len(points)} точек. Она уже применяется к видеопотоку."


@lru_cache(maxsize=1)
def get_model():
    return YOLO(MODEL_NAME)


@spaces.GPU(duration=10)
def analyze_frame(frame_rgb, confidence, selected, profiles):
    if frame_rgb is None:
        return None, "### Камера не подключена\nРазрешите браузеру доступ к камере."

    normalized_zone = (profiles or DEFAULT_CAMERAS).get(selected, [])
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    height, width = frame_bgr.shape[:2]
    polygon = denormalize_polygon(normalized_zone, width, height)

    if polygon:
        polygon_np = np.array(polygon, dtype=np.int32)
        for number, point in enumerate(polygon, start=1):
            cv2.circle(frame_bgr, point, 6, (0, 190, 255), -1)
            cv2.putText(frame_bgr, str(number), (point[0] + 7, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 190, 255), 2)
        if len(polygon) >= 3:
            overlay = frame_bgr.copy()
            cv2.fillPoly(overlay, [polygon_np], (0, 190, 255))
            frame_bgr = cv2.addWeighted(overlay, 0.16, frame_bgr, 0.84, 0)
            cv2.polylines(frame_bgr, [polygon_np], True, (0, 190, 255), 2)
        elif len(polygon) >= 2:
            cv2.polylines(frame_bgr, [polygon_np], False, (0, 190, 255), 2)

    with MODEL_LOCK:
        result = get_model().predict(
            frame_bgr, classes=[0], conf=float(confidence), imgsz=640,
            device=0 if torch.cuda.is_available() else "cpu", verbose=False,
        )[0]

    people = 0
    violations = 0
    zone_ready = len(polygon) >= 3
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        foot = ((x1 + x2) // 2, y2)
        in_zone = zone_ready and point_in_polygon(foot, polygon)
        people += 1
        violations += int(in_zone)
        color = (0, 0, 235) if in_zone else (40, 190, 80)
        label = "DANGER" if in_zone else "PERSON"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame_bgr, foot, 5, color, -1)
        cv2.putText(frame_bgr, f"{label} {float(box.conf[0]):.0%}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

    if not zone_ready:
        status = f"### 🟡 {selected}\nЗона не задана: нажмите минимум в 3 местах на изображении результата."
    elif violations:
        status = f"### 🔴 {selected}: ТРЕВОГА\nВ опасной зоне: **{violations}** · Всего людей: **{people}**"
    else:
        status = f"### 🟢 {selected}: мониторинг\nЛюдей: **{people}** · Нарушений: **0**"
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), status


with gr.Blocks(title="Safety Vision") as demo:
    profiles = gr.State(DEFAULT_CAMERAS)
    frozen_frame = gr.State(None)
    gr.Markdown("# Safety Vision\n**Несколько камер · отдельная многоугольная зона для каждой.**")
    with gr.Row():
        with gr.Column(scale=1):
            selected_camera = gr.Dropdown(choices=list(DEFAULT_CAMERAS), value="Камера 1", label="Текущая камера")
            with gr.Row():
                camera_name = gr.Textbox(label="Название новой камеры", placeholder="Например: Склад — ворота")
                add_camera_button = gr.Button("Добавить")
            delete_camera_button = gr.Button("Удалить текущую камеру", size="sm")
            camera = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Источник выбранной камеры")
            confidence = gr.Slider(0.2, 0.9, value=0.45, step=0.05, label="Порог уверенности")
            freeze_button = gr.Button("Зафиксировать кадр для разметки", variant="primary")
        with gr.Column(scale=1):
            output = gr.Image(label="Мониторинг", streaming=True, interactive=False)
            zone_canvas = gr.Image(
                label="Редактор зоны — ставьте вершины кликами",
                type="numpy",
                interactive=True,
                visible=True,
            )
            with gr.Row():
                undo_button = gr.Button("Отменить точку")
                clear_button = gr.Button("Очистить зону")
                save_button = gr.Button("Сохранить зону", variant="primary")
            zone_status = gr.Markdown("Зона **Камера 1**: 0 точек — нужно минимум 3.")
            status = gr.Markdown("### Ожидание камеры")

    add_camera_button.click(add_camera, [camera_name, profiles], [profiles, selected_camera, zone_status])
    delete_camera_button.click(delete_camera, [selected_camera, profiles], [profiles, selected_camera, zone_status])
    freeze_button.click(freeze_frame, [camera, selected_camera, profiles], [frozen_frame, zone_canvas, zone_status])
    undo_button.click(undo_point, [selected_camera, profiles, frozen_frame], [profiles, zone_canvas, zone_status])
    clear_button.click(clear_zone, [selected_camera, profiles, frozen_frame], [profiles, zone_canvas, zone_status])
    save_button.click(save_zone, [selected_camera, profiles], zone_status)
    selected_camera.change(lambda name, data: zone_message(name, (data or {}).get(name, [])), [selected_camera, profiles], zone_status)
    zone_canvas.select(add_zone_point, [selected_camera, profiles, frozen_frame], [profiles, zone_canvas, zone_status], show_progress="hidden")
    camera.stream(
        fn=analyze_frame,
        inputs=[camera, confidence, selected_camera, profiles], outputs=[output, status],
        stream_every=0.5, time_limit=600, concurrency_limit=1, api_name="analyze_frame",
    )
    gr.Markdown(
        "> Настройка хранится в текущей сессии браузера. Для промышленной версии нужны база камер, авторизация и локальный RTSP-шлюз.  \n"
        "> Это демонстрационный MVP, а не сертифицированная система безопасности. Кадры передаются в облако."
    )


if __name__ == "__main__":
    demo.launch()
