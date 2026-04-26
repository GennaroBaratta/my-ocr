"""OCR results screen composition."""

from __future__ import annotations

from pathlib import Path

import flet as ft

from ... import theme
from ...components.stepper import build_stepper
from ...ocr_result_text import current_page_ocr_markdown_for_state, ocr_json_text_for_state
from ...state import AppState
from .actions import ResultsScreenActions, page_label_text
from .presenter import build_results_split_pane
from .toolbar import build_results_toolbar


def build_results_view(
    page: ft.Page,
    state: AppState,
    file_picker: ft.FilePicker,
) -> ft.View:
    filename = Path(state.session.current_input_path).name if state.session.current_input_path else ""
    if not filename:
        filename = state.session.run_id or ""

    content_host = ft.Row(spacing=0, expand=True)
    document_viewer_width = 500.0

    page_label = ft.Text(
        page_label_text(state),
        size=13,
        color=theme.TEXT_PRIMARY,
        width=80,
        text_align=ft.TextAlign.CENTER,
    )

    def current_ocr_json_text() -> str:
        return ocr_json_text_for_state(state)

    def current_ocr_markdown_text() -> str:
        return state.session.ocr_markdown

    def current_page_export_markdown_text() -> str:
        return current_page_ocr_markdown_for_state(state)

    def update_document_width(width: float) -> None:
        nonlocal document_viewer_width
        document_viewer_width = width

    def build_content_split_pane() -> ft.Control:
        return build_results_split_pane(
            state,
            document_viewer_width=document_viewer_width,
            on_document_width_change=update_document_width,
        )

    def rebuild() -> None:
        page_label.value = page_label_text(state)
        actions.sync_toolbar_state()
        content_host.controls = [build_content_split_pane()]
        page.update()

    actions = ResultsScreenActions(
        page,
        state,
        file_picker,
        rebuild=rebuild,
        current_ocr_json_text=current_ocr_json_text,
        current_ocr_markdown_text=current_ocr_markdown_text,
        current_page_export_markdown_text=current_page_export_markdown_text,
    )
    toolbar, toolbar_controls = build_results_toolbar(
        page,
        state,
        filename=filename,
        actions=actions,
        page_label=page_label,
    )
    actions.bind_toolbar_controls(toolbar_controls)

    content_host.controls = [build_content_split_pane()]

    return ft.View(
        route=f"/results/{state.session.run_id}",
        controls=[
            ft.Column(
                [
                    build_stepper(3, page, state),
                    ft.Column([toolbar, content_host], spacing=0, expand=True),
                ],
                spacing=0,
                expand=True,
            )
        ],
        bgcolor=theme.BG_PAGE,
        padding=0,
    )
