"""Interactive bounding-box editor on a document page image."""

from __future__ import annotations

from typing import Callable

import flet as ft
from flet import Alignment, BoxFit

from .. import theme
from ..image_utils import get_image_size
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
            alignment=Alignment.CENTER,
            expand=True,
            bgcolor=theme.BG_PAGE,
        )

    scale = state.zoom_level
    image_width, image_height = get_image_size(page_data.image_path)
    canvas_width = max(1, int(image_width * scale)) if image_width else None
    canvas_height = max(1, int(image_height * scale)) if image_height else None
    image = ft.Image(
        src=page_data.image_path,
        width=canvas_width,
        height=canvas_height,
        fit=BoxFit.FILL if canvas_width and canvas_height else BoxFit.CONTAIN,
    )

    overlays: list[ft.Control] = []
    for box in page_data.boxes:
        overlays.extend(
            _build_box_overlay(
                box,
                scale,
                image_width,
                image_height,
                on_box_selected,
                on_box_changed,
            )
        )

    stack = ft.Stack(
        [image, *overlays],
        width=canvas_width,
        height=canvas_height,
    )

    canvas = ft.Column(
        [
            ft.Row(
                [stack],
                alignment=ft.MainAxisAlignment.CENTER,
                scroll=ft.ScrollMode.AUTO,
            )
        ],
        expand=True,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        scroll=ft.ScrollMode.AUTO,
    )

    return ft.Container(
        content=canvas,
        expand=True,
        bgcolor=theme.BG_PAGE,
        padding=16,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )


def _build_box_overlay(
    box: BoundingBox,
    scale: float,
    max_width: int,
    max_height: int,
    on_select: Callable[[str | None], None],
    on_change: Callable[[], None],
) -> list[ft.Control]:
    is_sel = box.selected
    color = theme.BOX_SELECTED if is_sel else theme.BOX_UNSELECTED

    def on_tap(e: ft.TapEvent) -> None:
        on_select(box.id)

    def on_drag(e: ft.DragUpdateEvent) -> None:
        delta = e.local_delta or e.global_delta
        if delta is None:
            return
        box.x += delta.x / scale
        box.y += delta.y / scale
        _clamp_box(box, max_width, max_height)
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
        controls.extend(_build_resize_handles(box, scale, max_width, max_height, on_change))

    return controls


def _build_resize_handles(
    box: BoundingBox,
    scale: float,
    max_width: int,
    max_height: int,
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
                delta = e.local_delta or e.global_delta
                if delta is None:
                    return
                dx = delta.x / scale
                dy = delta.y / scale
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
                _clamp_box(box, max_width, max_height)
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


def _clamp_box(box: BoundingBox, max_width: int, max_height: int) -> None:
    box.width = max(10, box.width)
    box.height = max(10, box.height)

    if max_width > 0:
        box.width = min(box.width, max_width)
        box.x = min(max(box.x, 0), max_width - box.width)

    if max_height > 0:
        box.height = min(box.height, max_height)
        box.y = min(max(box.y, 0), max_height - box.height)
