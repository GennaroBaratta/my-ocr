"""Read-only document image viewer with bounding-box overlays and zoom."""

from __future__ import annotations

import flet as ft
from flet import BoxFit

from .. import theme
from ..image_utils import get_image_size
from ..state import AppState, PageData
from .overlay_styles import overlay_colors_for_label


def build_doc_viewer(state: AppState) -> ft.Column:
    page_data = state.current_page
    if not page_data:
        return ft.Column(
            [
                ft.Icon(
                    ft.Icons.IMAGE_NOT_SUPPORTED,
                    size=48,
                    color=theme.TEXT_MUTED,
                ),
                ft.Text("No pages available", color=theme.TEXT_MUTED),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        )

    zoom_text = ft.Text(
        f"{int(state.zoom_level * 100)}%",
        size=12,
        color=theme.TEXT_PRIMARY,
        width=40,
        text_align=ft.TextAlign.CENTER,
    )

    def zoom_in(event_page: ft.Page | ft.BasePage) -> None:
        state.zoom_level = min(3.0, state.zoom_level + 0.25)
        zoom_text.value = f"{int(state.zoom_level * 100)}%"
        _rebuild_canvas(canvas_stack, page_data, state)
        event_page.update()

    def zoom_out(event_page: ft.Page | ft.BasePage) -> None:
        state.zoom_level = max(0.25, state.zoom_level - 0.25)
        zoom_text.value = f"{int(state.zoom_level * 100)}%"
        _rebuild_canvas(canvas_stack, page_data, state)
        event_page.update()

    header = ft.Row(
        [
            ft.Text(
                "SOURCE DOCUMENT",
                size=11,
                weight=ft.FontWeight.W_600,
                color=theme.TEXT_MUTED,
            ),
            ft.Container(expand=True),
            ft.IconButton(
                icon=ft.Icons.REMOVE,
                icon_size=16,
                on_click=lambda e: zoom_out(e.page),
                icon_color=theme.TEXT_MUTED,
                tooltip="Zoom out",
            ),
            zoom_text,
            ft.IconButton(
                icon=ft.Icons.ADD,
                icon_size=16,
                on_click=lambda e: zoom_in(e.page),
                icon_color=theme.TEXT_MUTED,
                tooltip="Zoom in",
            ),
        ],
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        height=36,
    )

    canvas_stack = ft.Stack()
    _rebuild_canvas(canvas_stack, page_data, state)

    canvas = ft.Column(
        [
            ft.Row(
                [canvas_stack],
                alignment=ft.MainAxisAlignment.CENTER,
                scroll=ft.ScrollMode.AUTO,
            )
        ],
        expand=True,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        scroll=ft.ScrollMode.AUTO,
    )

    return ft.Column(
        [
            ft.Container(
                content=header,
                padding=ft.Padding.symmetric(horizontal=12, vertical=4),
                bgcolor=theme.BG_SURFACE,
                border=ft.Border.only(bottom=ft.BorderSide(1, theme.BORDER)),
            ),
            ft.Container(
                content=canvas,
                expand=True,
                bgcolor=theme.BG_PAGE,
                padding=16,
            ),
        ],
        spacing=0,
        expand=True,
    )


def _rebuild_canvas(stack: ft.Stack, page_data: PageData, state: AppState) -> None:
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
        is_sel = box.selected
        color, fill = overlay_colors_for_label(box.label, is_sel)
        overlays.append(
            ft.Container(
                left=box.x * scale,
                top=box.y * scale,
                width=box.width * scale,
                height=box.height * scale,
                border=ft.Border.all(2 if is_sel else 1, color),
                bgcolor=fill,
            )
        )

    stack.width = canvas_width
    stack.height = canvas_height
    stack.controls = [image, *overlays]
