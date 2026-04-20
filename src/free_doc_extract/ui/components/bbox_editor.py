"""Interactive bounding-box editor on a document page image."""

from __future__ import annotations

from typing import Callable, cast

import flet as ft
from flet import Alignment, BoxFit

from .inspector import LABEL_TO_BLOCK_TYPE
from .. import theme
from ..image_utils import get_image_size
from ..state import AppState, BoundingBox


_BOX_KIND_COLORS = {
    "Text Block": theme.BOX_TEXT_BLOCK,
    "Table": theme.BOX_TABLE,
    "Figure/Image": theme.BOX_FIGURE_IMAGE,
    "Header": theme.BOX_HEADER,
    "Title": theme.BOX_HEADER,
    "Formula": theme.BOX_FORMULA,
}

_UNSELECTED_BORDER_ALPHA = "14"
_UNSELECTED_FILL_ALPHA = "02"
_MOVE_HANDLE_SIZE = 16


def build_bbox_editor(
    state: AppState,
    on_box_selected: Callable[[str | None], None],
    on_box_changed: Callable[[], None],
    on_box_live_change: Callable[[], None],
) -> ft.Container:
    stack = _build_editor_stack(state, on_box_selected, on_box_changed, on_box_live_change)
    if stack is None:
        return ft.Container(
            content=ft.Text("No page loaded", color=theme.TEXT_MUTED),
            alignment=Alignment.CENTER,
            expand=True,
            bgcolor=theme.BG_PAGE,
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


def refresh_bbox_editor(
    editor: ft.Container,
    state: AppState,
    on_box_selected: Callable[[str | None], None],
    on_box_changed: Callable[[], None],
    on_box_live_change: Callable[[], None],
) -> None:
    stack = _build_editor_stack(state, on_box_selected, on_box_changed, on_box_live_change)
    if stack is None or not isinstance(editor.content, ft.Column):
        replacement = build_bbox_editor(state, on_box_selected, on_box_changed, on_box_live_change)
        editor.content = replacement.content
        editor.alignment = replacement.alignment
        editor.padding = replacement.padding
        return

    stack_row = cast(ft.Row, editor.content.controls[0])
    stack_row.controls = [stack]


def _build_editor_stack(
    state: AppState,
    on_box_selected: Callable[[str | None], None],
    on_box_changed: Callable[[], None],
    on_box_live_change: Callable[[], None],
) -> ft.Stack | None:
    page_data = state.current_page
    if not page_data:
        return None

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
                on_box_live_change,
            )
        )

    return ft.Stack(
        [image, *overlays],
        width=canvas_width,
        height=canvas_height,
    )


def _build_box_overlay(
    box: BoundingBox,
    scale: float,
    max_width: int,
    max_height: int,
    on_select: Callable[[str | None], None],
    on_change: Callable[[], None],
    on_live_change: Callable[[], None],
) -> list[ft.Control]:
    is_sel = box.selected
    color, fill = _overlay_colors(box)

    box_content = ft.Container(
        width=box.width * scale,
        height=box.height * scale,
        border=ft.Border.all(2 if is_sel else 1, color),
        bgcolor=fill,
        tooltip=box.label,
    )

    handles: list[ft.GestureDetector] = []
    move_handle_ref: list[ft.GestureDetector] = []

    def refresh_overlay() -> None:
        _sync_box_overlay(box_container, box_content, handles, box, scale)
        if move_handle_ref:
            mx, my = _move_handle_position(box, scale)
            move_handle_ref[0].left = mx
            move_handle_ref[0].top = my

    def on_tap(e: ft.TapEvent) -> None:
        on_select(box.id)

    def on_move(e: ft.DragUpdateEvent) -> None:
        delta = e.local_delta or e.global_delta
        if delta is None:
            return
        box.x += delta.x / scale
        box.y += delta.y / scale
        _clamp_box(box, max_width, max_height)
        refresh_overlay()
        on_live_change()

    def on_move_end(e: ft.DragEndEvent) -> None:
        on_change()

    box_container = ft.GestureDetector(
        content=box_content,
        mouse_cursor=ft.MouseCursor.MOVE if is_sel else ft.MouseCursor.CLICK,
        on_tap=on_tap,
        left=box.x * scale,
        top=box.y * scale,
    )

    overlays: list[ft.Control] = [box_container]

    if is_sel:
        handles = _build_resize_handles(
            box,
            scale,
            max_width,
            max_height,
            on_change,
            on_live_change,
            refresh_overlay,
        )
        move_handle = _build_move_handle(box, scale, on_move, on_move_end)
        move_handle_ref.append(move_handle)
        overlays.extend(handles)
        overlays.append(move_handle)

    return overlays


def _build_resize_handles(
    box: BoundingBox,
    scale: float,
    max_width: int,
    max_height: int,
    on_change: Callable[[], None],
    on_live_change: Callable[[], None],
    refresh_overlay: Callable[[], None],
) -> list[ft.GestureDetector]:
    handle_size = 8
    handles: list[ft.GestureDetector] = []
    for left, top, cursor, mode in _handle_layouts(box, scale, handle_size):

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
                refresh_overlay()
                on_live_change()

            return handler

        handles.append(
            ft.GestureDetector(
                content=ft.Container(
                    width=handle_size,
                    height=handle_size,
                    bgcolor=theme.PRIMARY,
                    border=ft.Border.all(1, "white"),
                    border_radius=1,
                ),
                mouse_cursor=cursor,
                on_pan_update=make_handler(mode),
                on_pan_end=lambda e: on_change(),
                left=left,
                top=top,
            )
        )

    return handles


def _move_handle_position(box: BoundingBox, scale: float) -> tuple[float, float]:
    return box.x * scale + 2, box.y * scale + 2


def _build_move_handle(
    box: BoundingBox,
    scale: float,
    on_move: Callable[[ft.DragUpdateEvent], None],
    on_move_end: Callable[[ft.DragEndEvent], None],
) -> ft.GestureDetector:
    left, top = _move_handle_position(box, scale)
    return ft.GestureDetector(
        content=ft.Container(
            width=_MOVE_HANDLE_SIZE,
            height=_MOVE_HANDLE_SIZE,
            bgcolor=theme.PRIMARY,
            border=ft.Border.all(1, "white"),
            border_radius=3,
            content=ft.Icon(
                ft.Icons.OPEN_WITH,
                size=_MOVE_HANDLE_SIZE - 4,
                color="white",
            ),
            alignment=ft.Alignment.CENTER,
            tooltip="Drag to move",
        ),
        mouse_cursor=ft.MouseCursor.MOVE,
        on_pan_update=on_move,
        on_pan_end=on_move_end,
        left=left,
        top=top,
    )


def _sync_box_overlay(
    box_container: ft.GestureDetector,
    box_content: ft.Container,
    handles: list[ft.GestureDetector],
    box: BoundingBox,
    scale: float,
) -> None:
    box_container.left = box.x * scale
    box_container.top = box.y * scale
    box_content.width = box.width * scale
    box_content.height = box.height * scale

    for handle, (left, top, _, _) in zip(handles, _handle_layouts(box, scale, 8), strict=False):
        handle.left = left
        handle.top = top


def _handle_layouts(
    box: BoundingBox,
    scale: float,
    handle_size: int,
) -> list[tuple[float, float, ft.MouseCursor, str]]:
    hs = handle_size / 2
    bx, by = box.x * scale, box.y * scale
    bw, bh = box.width * scale, box.height * scale
    return [
        (bx - hs, by - hs, ft.MouseCursor.RESIZE_UP_LEFT, "x_y_wh_neg"),
        (bx + bw - hs, by - hs, ft.MouseCursor.RESIZE_UP_RIGHT, "y_w_neg_h"),
        (bx - hs, by + bh - hs, ft.MouseCursor.RESIZE_DOWN_LEFT, "x_w_neg_h"),
        (bx + bw - hs, by + bh - hs, ft.MouseCursor.RESIZE_DOWN_RIGHT, "wh"),
        (bx + bw / 2 - hs, by - hs, ft.MouseCursor.RESIZE_UP, "y_h_neg"),
        (bx + bw / 2 - hs, by + bh - hs, ft.MouseCursor.RESIZE_DOWN, "h_only"),
        (bx - hs, by + bh / 2 - hs, ft.MouseCursor.RESIZE_LEFT, "x_w_neg"),
        (bx + bw - hs, by + bh / 2 - hs, ft.MouseCursor.RESIZE_RIGHT, "w_only"),
    ]


def _clamp_box(box: BoundingBox, max_width: int, max_height: int) -> None:
    box.width = max(10, box.width)
    box.height = max(10, box.height)

    if max_width > 0:
        box.width = min(box.width, max_width)
        box.x = min(max(box.x, 0), max_width - box.width)

    if max_height > 0:
        box.height = min(box.height, max_height)
        box.y = min(max(box.y, 0), max_height - box.height)


def _overlay_colors(box: BoundingBox) -> tuple[str, str]:
    block_type = LABEL_TO_BLOCK_TYPE.get(box.label, "Text Block")
    base_color = _BOX_KIND_COLORS.get(block_type, theme.BOX_TEXT_BLOCK)
    if box.selected:
        return base_color, f"{base_color}1A"
    return (
        f"{base_color}{_UNSELECTED_BORDER_ALPHA}",
        f"{base_color}{_UNSELECTED_FILL_ALPHA}",
    )
