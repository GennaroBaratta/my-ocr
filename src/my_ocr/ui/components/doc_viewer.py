"""Read-only document image viewer with bounding-box overlays and zoom."""

from __future__ import annotations

import flet as ft
from flet import BoxFit

from .. import theme
from ..image_utils import get_image_size, get_image_source
from ..state import AppState, PageData
from ..zoom import (
    ZOOM_MODE_FIT_WIDTH,
    effective_zoom_level,
    set_manual_zoom,
    set_zoom_available_width,
    toggle_fit_width_zoom,
    zoom_label_text,
)
from .overlay_styles import overlay_colors_for_label

_SET_AVAILABLE_WIDTH_ATTR = "_my_ocr_set_available_width"


def build_doc_viewer(state: AppState, available_width: float | None = None) -> ft.Column:
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

    if available_width is not None:
        set_zoom_available_width(state, available_width)

    zoom_text = ft.Text(
        "",
        size=12,
        color=theme.TEXT_PRIMARY,
        width=64,
        text_align=ft.TextAlign.CENTER,
    )
    fit_width_button = ft.IconButton(
        icon=ft.Icons.WIDTH_FULL,
        icon_size=16,
        icon_color=theme.PRIMARY
        if state.session.zoom_mode == ZOOM_MODE_FIT_WIDTH
        else theme.TEXT_MUTED,
        on_click=lambda e: fit_width(e.page),
        tooltip="Fit page width",
    )

    def refresh_canvas() -> None:
        scale = _rebuild_canvas(canvas_stack, page_data, state)
        zoom_text.value = zoom_label_text(state, scale)
        fit_width_button.icon_color = (
            theme.PRIMARY if state.session.zoom_mode == ZOOM_MODE_FIT_WIDTH else theme.TEXT_MUTED
        )

    def zoom_in(event_page: ft.Page | ft.BasePage) -> None:
        set_manual_zoom(state, _current_scale(page_data, state) + 0.25)
        refresh_canvas()
        event_page.update()

    def zoom_out(event_page: ft.Page | ft.BasePage) -> None:
        set_manual_zoom(state, _current_scale(page_data, state) - 0.25)
        refresh_canvas()
        event_page.update()

    def fit_width(event_page: ft.Page | ft.BasePage) -> None:
        image_width, _image_height = get_image_size(page_data.image_path)
        toggle_fit_width_zoom(state, image_width)
        refresh_canvas()
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
            fit_width_button,
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
    refresh_canvas()

    def refresh_available_width(width: float) -> None:
        set_zoom_available_width(state, width)
        refresh_canvas()
        controls_to_update: list[ft.Control] = [zoom_text, fit_width_button]
        for control in controls_to_update:
            try:
                control.update()
            except RuntimeError:
                pass

    def on_viewer_size_change(e: ft.PageResizeEvent) -> None:
        refresh_available_width(e.width)
        controls_to_update: list[ft.Control] = []
        try:
            controls_to_update.append(e.control)
        except AttributeError:
            pass
        for control in controls_to_update:
            try:
                control.update()
            except RuntimeError:
                pass

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

    viewer = ft.Column(
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
                on_size_change=on_viewer_size_change,
            ),
        ],
        spacing=0,
        expand=True,
    )
    setattr(viewer, _SET_AVAILABLE_WIDTH_ATTR, refresh_available_width)
    return viewer


def refresh_doc_viewer_available_width(viewer: ft.Control, width: float) -> None:
    refresh = getattr(viewer, _SET_AVAILABLE_WIDTH_ATTR, None)
    if callable(refresh):
        refresh(width)


def _current_scale(page_data: PageData, state: AppState) -> float:
    image_width, image_height = get_image_size(page_data.image_path)
    _ = image_height
    return effective_zoom_level(state, image_width)


def _rebuild_canvas(stack: ft.Stack, page_data: PageData, state: AppState) -> float:
    image_width, image_height = get_image_size(page_data.image_path)
    scale = effective_zoom_level(state, image_width)
    canvas_width = max(1, int(image_width * scale)) if image_width else None
    canvas_height = max(1, int(image_height * scale)) if image_height else None
    image = ft.Image(
        src=get_image_source(page_data.image_path),
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
    return scale

