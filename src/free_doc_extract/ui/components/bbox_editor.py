"""Interactive bounding-box editor on a document page image."""

from __future__ import annotations

from typing import Callable

import flet as ft

from .. import theme
from ..state import AppState, BoundingBox


def build_bbox_editor(
    state: AppState,
    on_box_selected: Callable[[str | None], None],
    on_box_changed: Callable[[], None],
) -> ft.Container:
    page_data = state.current_page
    if not page_data:
        return ft.Container(
            content=ft.Text("No page loaded", color=theme.TEXT_MUTED),
            alignment=ft.alignment.center,
            expand=True,
            bgcolor=theme.BG_PAGE,
        )

    scale = state.zoom_level
    image = ft.Image(
        src=page_data.image_path,
        fit=ft.ImageFit.CONTAIN,
    )

    overlays: list[ft.Control] = []
    for box in page_data.boxes:
        overlays.extend(
            _build_box_overlay(box, scale, on_box_selected, on_box_changed)
        )

    stack = ft.Stack(
        [image, *overlays],
        expand=True,
    )

    return ft.Container(
        content=stack,
        expand=True,
        alignment=ft.alignment.center,
        bgcolor=theme.BG_PAGE,
        padding=16,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )


def _build_box_overlay(
    box: BoundingBox,
    scale: float,
    on_select: Callable[[str | None], None],
    on_change: Callable[[], None],
) -> list[ft.Control]:
    is_sel = box.selected
    color = theme.BOX_SELECTED if is_sel else theme.BOX_UNSELECTED

    def on_tap(e: ft.TapEvent) -> None:
        on_select(box.id)

    def on_drag(e: ft.DragUpdateEvent) -> None:
        box.x += e.delta_x / scale
        box.y += e.delta_y / scale
        on_change()

    box_container = ft.GestureDetector(
        content=ft.Container(
            width=box.width * scale,
            height=box.height * scale,
            border=ft.border.all(2 if is_sel else 1, color),
            bgcolor=f"{color}1A",
            tooltip=box.label,
        ),
        mouse_cursor=ft.MouseCursor.MOVE if is_sel else ft.MouseCursor.CLICK,
        on_tap=on_tap,
        on_pan_update=on_drag if is_sel else None,
        left=box.x * scale,
        top=box.y * scale,
    )

    controls: list[ft.Control] = [box_container]

    # Resize handles for selected box
    if is_sel:
        controls.extend(_build_resize_handles(box, scale, on_change))

    return controls


def _build_resize_handles(
    box: BoundingBox,
    scale: float,
    on_change: Callable[[], None],
) -> list[ft.Control]:
    handle_size = 8
    hs = handle_size / 2
    bx, by = box.x * scale, box.y * scale
    bw, bh = box.width * scale, box.height * scale

    # (position_left, position_top, cursor, dx_affects, dy_affects)
    handle_defs = [
        # corners
        (bx - hs, by - hs, ft.MouseCursor.RESIZE_UP_LEFT, "x_y_wh_neg"),
        (bx + bw - hs, by - hs, ft.MouseCursor.RESIZE_UP_RIGHT, "y_w_neg_h"),
        (bx - hs, by + bh - hs, ft.MouseCursor.RESIZE_DOWN_LEFT, "x_w_neg_h"),
        (bx + bw - hs, by + bh - hs, ft.MouseCursor.RESIZE_DOWN_RIGHT, "wh"),
        # edges
        (bx + bw / 2 - hs, by - hs, ft.MouseCursor.RESIZE_UP, "y_h_neg"),
        (bx + bw / 2 - hs, by + bh - hs, ft.MouseCursor.RESIZE_DOWN, "h_only"),
        (bx - hs, by + bh / 2 - hs, ft.MouseCursor.RESIZE_LEFT, "x_w_neg"),
        (bx + bw - hs, by + bh / 2 - hs, ft.MouseCursor.RESIZE_RIGHT, "w_only"),
    ]

    handles: list[ft.Control] = []
    for left, top, cursor, mode in handle_defs:

        def make_handler(m: str):  # noqa: E301
            def handler(e: ft.DragUpdateEvent) -> None:
                dx = e.delta_x / scale
                dy = e.delta_y / scale
                if m == "x_y_wh_neg":
                    box.x += dx
                    box.y += dy
                    box.width -= dx
                    box.height -= dy
                elif m == "y_w_neg_h":
                    box.y += dy
                    box.width += dx
                    box.height -= dy
                elif m == "x_w_neg_h":
                    box.x += dx
                    box.width -= dx
                    box.height += dy
                elif m == "wh":
                    box.width += dx
                    box.height += dy
                elif m == "y_h_neg":
                    box.y += dy
                    box.height -= dy
                elif m == "h_only":
                    box.height += dy
                elif m == "x_w_neg":
                    box.x += dx
                    box.width -= dx
                elif m == "w_only":
                    box.width += dx
                # Enforce minimums
                box.width = max(10, box.width)
                box.height = max(10, box.height)
                on_change()
            return handler

        handles.append(
            ft.GestureDetector(
                content=ft.Container(
                    width=handle_size,
                    height=handle_size,
                    bgcolor=theme.PRIMARY,
                    border=ft.border.all(1, "white"),
                    border_radius=1,
                ),
                mouse_cursor=cursor,
                on_pan_update=make_handler(mode),
                left=left,
                top=top,
            )
        )

    return handles
