"""Read-only document image viewer with bounding-box overlays and zoom."""

from __future__ import annotations

import flet as ft

from .. import theme
from ..state import AppState, PageData


def build_doc_viewer(state: AppState) -> ft.Column:
    page_data = state.current_page
    if not page_data:
        return ft.Column(
            [ft.Text("No pages available", color=theme.TEXT_MUTED)],
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

    def zoom_in(e: ft.ControlEvent) -> None:
        state.zoom_level = min(3.0, state.zoom_level + 0.25)
        zoom_text.value = f"{int(state.zoom_level * 100)}%"
        _rebuild_canvas(canvas_stack, page_data, state)
        e.page.update()

    def zoom_out(e: ft.ControlEvent) -> None:
        state.zoom_level = max(0.25, state.zoom_level - 0.25)
        zoom_text.value = f"{int(state.zoom_level * 100)}%"
        _rebuild_canvas(canvas_stack, page_data, state)
        e.page.update()

    header = ft.Row(
        [
            ft.Text(
                "SOURCE DOCUMENT",
                size=11,
                weight=ft.FontWeight.W_600,
                color=theme.TEXT_MUTED,
                letter_spacing=1.2,
            ),
            ft.Container(expand=True),
            ft.IconButton(
                icon=ft.Icons.REMOVE, icon_size=16, on_click=zoom_out,
                icon_color=theme.TEXT_MUTED, tooltip="Zoom out",
            ),
            zoom_text,
            ft.IconButton(
                icon=ft.Icons.ADD, icon_size=16, on_click=zoom_in,
                icon_color=theme.TEXT_MUTED, tooltip="Zoom in",
            ),
        ],
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        height=36,
    )

    canvas_stack = ft.Stack(expand=True)
    _rebuild_canvas(canvas_stack, page_data, state)

    return ft.Column(
        [
            ft.Container(
                content=header,
                padding=ft.padding.symmetric(horizontal=12, vertical=4),
                bgcolor=theme.BG_SURFACE,
                border=ft.border.only(bottom=ft.BorderSide(1, theme.BORDER)),
            ),
            ft.Container(
                content=canvas_stack,
                expand=True,
                alignment=ft.alignment.center,
                bgcolor=theme.BG_PAGE,
                padding=16,
            ),
        ],
        spacing=0,
        expand=True,
    )


def _rebuild_canvas(
    stack: ft.Stack, page_data: PageData, state: AppState
) -> None:
    scale = state.zoom_level
    image = ft.Image(
        src=page_data.image_path,
        fit=ft.ImageFit.CONTAIN,
    )

    overlays: list[ft.Control] = []
    for box in page_data.boxes:
        is_sel = box.selected
        color = theme.BOX_SELECTED if is_sel else theme.BOX_UNSELECTED
        overlays.append(
            ft.Container(
                left=box.x * scale,
                top=box.y * scale,
                width=box.width * scale,
                height=box.height * scale,
                border=ft.border.all(2 if is_sel else 1, color),
                bgcolor=f"{color}1A",
            )
        )

    stack.controls = [image, *overlays]
