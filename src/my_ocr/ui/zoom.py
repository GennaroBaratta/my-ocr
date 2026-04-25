"""Shared document zoom helpers for review and results views."""

from __future__ import annotations

from typing import Any

ZOOM_MODE_FIT_WIDTH = "fit_width"
ZOOM_MODE_MANUAL = "manual"
ZOOM_MIN = 0.25
ZOOM_MAX = 3.0
VIEWER_HORIZONTAL_PADDING = 32


def clamp_zoom(value: float) -> float:
    return max(ZOOM_MIN, min(ZOOM_MAX, value))


def effective_zoom_level(state: Any, image_width: int | None) -> float:
    if getattr(state, "zoom_mode", ZOOM_MODE_MANUAL) == ZOOM_MODE_FIT_WIDTH:
        available_width = getattr(state, "zoom_fit_width", None)
        if image_width and available_width:
            content_width = max(1.0, float(available_width) - VIEWER_HORIZONTAL_PADDING)
            return clamp_zoom(content_width / image_width)
    return clamp_zoom(float(getattr(state, "zoom_level", 1.0)))


def set_manual_zoom(state: Any, value: float) -> float:
    state.zoom_mode = ZOOM_MODE_MANUAL
    state.zoom_level = clamp_zoom(value)
    return state.zoom_level


def set_fit_width_zoom(state: Any) -> None:
    state.zoom_mode = ZOOM_MODE_FIT_WIDTH


def set_zoom_available_width(state: Any, width: float | None) -> None:
    state.zoom_fit_width = max(0.0, float(width or 0))


def zoom_label_text(state: Any, scale: float) -> str:
    percent = f"{int(scale * 100)}%"
    if getattr(state, "zoom_mode", ZOOM_MODE_MANUAL) == ZOOM_MODE_FIT_WIDTH:
        return f"Fit {percent}"
    return percent
