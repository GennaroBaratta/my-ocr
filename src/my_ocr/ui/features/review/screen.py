"""Document review workspace — 3-pane layout with page strip, bbox editor, inspector."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import flet as ft

from ... import theme
from ...components.bbox_editor import refresh_bbox_editor
from ...components.inspector import build_inspector
from ...components.loading_overlay import build_loading_overlay
from ...components.stepper import build_stepper
from ...state import AppState
from .actions import (
    ReviewScreenActions,
    current_zoom_scale,
    fit_width_icon_color,
    review_page_label_text,
)
from .presenter import build_review_panes, sync_add_box_button
from .toolbar import build_add_box_button
from ...zoom import set_zoom_available_width, zoom_label_text


def build_review_view(page: ft.Page, state: AppState) -> ft.View:
    filename = Path(state.session.current_input_path).name if state.session.current_input_path else ""
    if not filename:
        filename = state.session.run_id or "Document"

    loading_overlay = build_loading_overlay("Running OCR...")

    content_row = ft.Row(spacing=0, expand=True)
    review_content_width: float | None = None

    def build_content_panes() -> list[ft.Control]:
        return build_review_panes(
            state,
            actions.select_page,
            actions.select_box,
            actions.box_changed,
            actions.box_live_changed,
            actions.zoom_scale_changed,
            actions.deselect_box,
            actions.remove_box,
        )

    def sync_editor_available_width(*, update: bool = False) -> bool:
        if review_content_width is None or len(content_row.controls) < 2:
            return False

        page_strip_width = float(getattr(content_row.controls[0], "width", 0) or 0)
        inspector_width = 0.0
        if len(content_row.controls) > 2:
            inspector_width = float(getattr(content_row.controls[2], "width", 0) or 0)

        set_zoom_available_width(
            state,
            max(0.0, review_content_width - page_strip_width - inspector_width),
        )
        bbox_editor = cast(ft.Container, content_row.controls[1])
        refresh_bbox_editor(
            bbox_editor,
            state,
            actions.select_box,
            actions.box_changed,
            actions.box_live_changed,
            actions.zoom_scale_changed,
        )
        if update:
            try:
                bbox_editor.update()
            except RuntimeError:
                pass
        return True

    def on_content_row_size_change(e: ft.LayoutSizeChangeEvent[ft.LayoutControl]) -> None:
        nonlocal review_content_width
        review_content_width = e.width
        sync_editor_available_width(update=True)

    content_row.on_size_change = on_content_row_size_change

    def rebuild() -> None:
        content_row.controls = build_content_panes()
        sync_editor_available_width()
        page.update()

    def refresh_selection() -> None:
        if len(content_row.controls) < 3:
            rebuild()
            return

        sync_add_box_button(state, add_box_label, add_box_btn, update=True)

        bbox_editor = cast(ft.Container, content_row.controls[1])
        refresh_bbox_editor(
            bbox_editor,
            state,
            actions.select_box,
            actions.box_changed,
            actions.box_live_changed,
            actions.zoom_scale_changed,
        )
        content_row.controls[2] = build_inspector(
            state,
            actions.deselect_box,
            actions.box_changed,
            actions.remove_box,
        )
        sync_editor_available_width()
        page.update()

    page_label = ft.Text(
        review_page_label_text(state),
        size=13,
        color=theme.TEXT_PRIMARY,
        width=80,
        text_align=ft.TextAlign.CENTER,
    )
    zoom_label = ft.Text(
        zoom_label_text(state, current_zoom_scale(state)),
        size=13,
        color=theme.TEXT_PRIMARY,
        width=64,
        text_align=ft.TextAlign.CENTER,
    )
    fit_width_btn = ft.IconButton(
        icon=ft.Icons.WIDTH_FULL,
        icon_size=16,
        icon_color=fit_width_icon_color(state, theme.PRIMARY, theme.TEXT_MUTED),
        tooltip="Fit page width",
    )

    def refresh_zoom_toolbar(scale: float | None = None) -> None:
        current_scale = scale if scale is not None else current_zoom_scale(state)
        zoom_label.value = zoom_label_text(state, current_scale)
        fit_width_btn.icon_color = fit_width_icon_color(state, theme.PRIMARY, theme.TEXT_MUTED)

    def update_zoom_toolbar_controls() -> None:
        for control in (zoom_label, fit_width_btn):
            try:
                control.update()
            except RuntimeError:
                pass

    def set_page_label(value: str) -> None:
        page_label.value = value

    def refresh_add_box_controls() -> None:
        sync_add_box_button(state, add_box_label, add_box_btn, update=True)

    actions = ReviewScreenActions(
        page,
        state,
        loading_overlay,
        rebuild=rebuild,
        refresh_selection=refresh_selection,
        refresh_add_box_controls=refresh_add_box_controls,
        set_page_label=set_page_label,
        refresh_zoom_toolbar=refresh_zoom_toolbar,
        update_zoom_toolbar_controls=update_zoom_toolbar_controls,
    )
    fit_width_btn.on_click = actions.toggle_fit_width

    pagination_controls: list[ft.Control] = [
        ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT,
            icon_size=18,
            icon_color=theme.TEXT_MUTED,
            on_click=actions.prev_page,
            tooltip="Previous page",
        ),
        page_label,
        ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT,
            icon_size=18,
            icon_color=theme.TEXT_MUTED,
            on_click=actions.next_page,
            tooltip="Next page",
        ),
    ]

    zoom_controls: list[ft.Control] = [
        fit_width_btn,
        ft.IconButton(
            icon=ft.Icons.REMOVE,
            icon_size=16,
            icon_color=theme.TEXT_MUTED,
            on_click=actions.zoom_out,
            tooltip="Zoom out",
        ),
        zoom_label,
        ft.IconButton(
            icon=ft.Icons.ADD,
            icon_size=16,
            icon_color=theme.TEXT_MUTED,
            on_click=actions.zoom_in,
            tooltip="Zoom in",
        ),
    ]

    add_box_label = ft.Text("Cancel Add" if state.session.is_adding_box else "Add Box")
    add_box_btn = build_add_box_button(state, add_box_label, actions.toggle_add_box)

    toolbar_controls: list[ft.Control] = [
        ft.IconButton(
            icon=ft.Icons.ARROW_BACK,
            icon_color=theme.TEXT_PRIMARY,
            icon_size=20,
            tooltip="Back",
            on_click=lambda: page.go("/"),
        ),
        ft.VerticalDivider(width=1, color=theme.BORDER),
        ft.Icon(ft.Icons.DESCRIPTION_OUTLINED, size=18, color=theme.TEXT_MUTED),
        ft.Text(
            filename,
            size=14,
            weight=ft.FontWeight.W_500,
            color=theme.TEXT_PRIMARY,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
            width=200,
        ),
        ft.Container(expand=True),
        ft.Container(
            content=ft.Row(
                pagination_controls,
                spacing=0,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            border=ft.Border.all(1, theme.BORDER),
            border_radius=6,
            bgcolor=theme.BG_ELEVATED,
            padding=ft.Padding.symmetric(horizontal=2),
        ),
        ft.Container(width=8),
        ft.Container(
            content=ft.Row(
                zoom_controls,
                spacing=0,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            border=ft.Border.all(1, theme.BORDER),
            border_radius=6,
            bgcolor=theme.BG_ELEVATED,
            padding=ft.Padding.symmetric(horizontal=2),
        ),
        ft.Container(width=8),
        ft.IconButton(
            icon=ft.Icons.REFRESH,
            icon_color=theme.TEXT_MUTED,
            icon_size=20,
            tooltip="Re-detect layout",
            on_click=actions.redetect_layout,
        ),
        add_box_btn,
        ft.Container(width=8),
        ft.Button(
            "Run OCR",
            icon=ft.Icons.PLAY_ARROW,
            on_click=actions.run_ocr,
            tooltip="Run OCR using the reviewed layout boxes",
            bgcolor=theme.PRIMARY,
            color="white",
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=6),
            ),
        ),
    ]

    toolbar = ft.Container(
        content=ft.Row(
            toolbar_controls,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=4,
        ),
        height=48,
        padding=ft.Padding.symmetric(horizontal=8),
        bgcolor=theme.BG_SURFACE,
        border=ft.Border.only(bottom=ft.BorderSide(1, theme.BORDER)),
    )

    page.on_keyboard_event = actions.keyboard

    content_row.controls = build_content_panes()

    return ft.View(
        route=f"/review/{state.session.run_id}",
        controls=[
            ft.Column(
                [
                    build_stepper(2, page, state),
                    ft.Stack(
                        [
                            ft.Column([toolbar, content_row], spacing=0, expand=True),
                            loading_overlay.control,
                        ],
                        expand=True,
                    ),
                ],
                spacing=0,
                expand=True,
            )
        ],
        bgcolor=theme.BG_PAGE,
        padding=0,
    )
