from __future__ import annotations

from typing import Iterable, Sequence


Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return True when a point is inside or on the boundary of a polygon."""
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if _on_segment(point, (xi, yi), (xj, yj)):
            return True
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersection:
                inside = not inside
        j = i
    return inside


def _on_segment(point: Point, a: Point, b: Point, epsilon: float = 1e-9) -> bool:
    px, py = point
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > epsilon:
        return False
    return min(ax, bx) - epsilon <= px <= max(ax, bx) + epsilon and min(ay, by) - epsilon <= py <= max(ay, by) + epsilon


def denormalize_polygon(points: Iterable[Sequence[float]], width: int, height: int) -> list[tuple[int, int]]:
    return [(round(float(x) * width), round(float(y) * height)) for x, y in points]


def normalize_polygon(points: Iterable[Sequence[int]], width: int, height: int) -> list[list[float]]:
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    return [[round(x / width, 6), round(y / height, 6)] for x, y in points]

