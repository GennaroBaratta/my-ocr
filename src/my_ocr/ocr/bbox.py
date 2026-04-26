from __future__ import annotations

from typing import Any

BOX_PADDING_X = 8
BOX_PADDING_Y = 6


def _sort_key_for_block(raw_bbox: Any) -> tuple[float, float, float, float]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    try:
        x1, y1, x2, y2 = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    return (y1, x1, y2, x2)


def normalize_bbox(
    raw_bbox: Any,
    width: int,
    height: int,
    *,
    coord_space: str | None = None,
) -> list[int] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return None

    if coord_space == "normalized":
        x1 = round(x1 * width / 1000)
        x2 = round(x2 * width / 1000)
        y1 = round(y1 * height / 1000)
        y2 = round(y2 * height / 1000)
    elif coord_space == "normalized_unit":
        x1 = round(x1 * width)
        x2 = round(x2 * width)
        y1 = round(y1 * height)
        y2 = round(y2 * height)
    elif coord_space == "pixel":
        x1, y1, x2, y2 = [round(value) for value in (x1, y1, x2, y2)]
    else:
        if max(x1, y1, x2, y2) <= 1:
            x1 = round(x1 * width)
            x2 = round(x2 * width)
            y1 = round(y1 * height)
            y2 = round(y2 * height)
        elif max(x1, y1, x2, y2) <= 1000 and (width > 1000 or height > 1000):
            x1 = round(x1 * width / 1000)
            x2 = round(x2 * width / 1000)
            y1 = round(y1 * height / 1000)
            y2 = round(y2 * height / 1000)
        else:
            x1, y1, x2, y2 = [round(value) for value in (x1, y1, x2, y2)]

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def pad_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    return [
        max(0, bbox[0] - BOX_PADDING_X),
        max(0, bbox[1] - BOX_PADDING_Y),
        min(width, bbox[2] + BOX_PADDING_X),
        min(height, bbox[3] + BOX_PADDING_Y),
    ]


def detect_bbox_coord_space(
    blocks: list[dict[str, Any]], *, width: int | None = None, height: int | None = None
) -> str:
    coords: list[float] = []
    per_axis: list[tuple[float, float, float, float]] = []
    for block in blocks:
        bbox = block.get("bbox_2d")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox]
            except (TypeError, ValueError):
                continue
            coords.extend((x1, y1, x2, y2))
            per_axis.append((x1, y1, x2, y2))
    if not coords:
        return "unknown"
    if all(0 <= value <= 1 for value in coords):
        return "normalized_unit"
    if any(value > 1000 for value in coords):
        return "pixel"
    if all(0 <= value <= 1000 for value in coords):
        return "normalized"
    if width is None or height is None:
        return "unknown"

    fits_pixel_bounds = all(
        0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height for x1, y1, x2, y2 in per_axis
    )
    exceeds_pixel_bounds = any(x2 > width or y2 > height for _, _, x2, y2 in per_axis)

    if exceeds_pixel_bounds and all(0 <= value <= 1000 for value in coords):
        return "normalized"
    if fits_pixel_bounds:
        return "pixel"
    return "unknown"
