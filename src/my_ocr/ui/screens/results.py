"""Results screen — split-pane with document viewer and code display."""

from __future__ import annotations

from pathlib import Path

import flet as ft

from .. import theme
from ..components.code_display import build_code_display
from ..components.doc_viewer import build_doc_viewer, refresh_doc_viewer_available_width
from ..components.split_pane import SplitPane
from ..components.stepper import build_stepper
from ..ocr_result_text import current_page_ocr_markdown_for_state, ocr_json_text_for_state
from ..state import AppState
from .results_actions import ResultsScreenActions, ResultsToolbarControls, page_label_text


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

    def build_content_split_pane() -> SplitPane:
        nonlocal document_viewer_width
        viewer = build_doc_viewer(state, available_width=document_viewer_width)

        def on_left_width_change(width: float) -> None:
            nonlocal document_viewer_width
            document_viewer_width = width
            refresh_doc_viewer_available_width(viewer, width)

        return SplitPane(
            viewer,
            build_code_display(state),
            initial_left_width=document_viewer_width,
            on_left_width_change=on_left_width_change,
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

    copy_json_button = ft.OutlinedButton(
        "Copy OCR JSON",
        icon=ft.Icons.CONTENT_COPY,
        tooltip="Copy OCR JSON to clipboard",
        on_click=actions.copy_clipboard,
        disabled=False,
        style=ft.ButtonStyle(
            color=theme.TEXT_PRIMARY,
            side=ft.BorderSide(1, theme.BORDER),
            shape=ft.RoundedRectangleBorder(radius=6),
        ),
    )
    download_page_markdown_button = ft.Button(
        "Download Page Markdown",
        icon=ft.Icons.DOWNLOAD,
        tooltip="Download OCR Markdown for this page",
        on_click=actions.download_page_markdown,
        disabled=False,
        bgcolor=theme.BG_ELEVATED,
        color=theme.TEXT_PRIMARY,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=6),
        ),
    )
    download_markdown_button = ft.Button(
        "Download OCR Markdown",
        icon=ft.Icons.DOWNLOAD,
        tooltip="Download OCR Markdown",
        on_click=actions.download_markdown,
        disabled=False,
        bgcolor=theme.BG_ELEVATED,
        color=theme.TEXT_PRIMARY,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=6),
        ),
    )
    download_json_button = ft.Button(
        "Download OCR JSON",
        icon=ft.Icons.DOWNLOAD,
        tooltip="Download OCR JSON",
        on_click=actions.download_json,
        disabled=False,
        bgcolor=theme.PRIMARY,
        color="white",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=6),
        ),
    )

    layout_rerun_button = ft.Button(
        "Re-detect This Page Layout",
        icon=ft.Icons.AUTO_FIX_HIGH,
        tooltip="Re-detect layout only for the active page",
        on_click=actions.rerun_page_layout,
        bgcolor=theme.BG_ELEVATED,
        color=theme.TEXT_PRIMARY,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=6),
        ),
    )
    ocr_rerun_button = ft.Button(
        "Re-run OCR For This Page",
        icon=ft.Icons.REFRESH,
        tooltip="Re-run OCR only for the active page",
        on_click=actions.rerun_page_ocr,
        bgcolor=theme.BG_ELEVATED,
        color=theme.TEXT_PRIMARY,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=6),
        ),
    )

    toolbar_controls: list[ft.Control] = [
        ft.IconButton(
            icon=ft.Icons.ARROW_BACK,
            icon_color=theme.TEXT_PRIMARY,
            icon_size=20,
            tooltip="Back to Layout Review",
            on_click=lambda: page.go(f"/review/{state.session.run_id}"),
        ),
        ft.VerticalDivider(width=1, color=theme.BORDER),
        ft.IconButton(
            icon=ft.Icons.RESTART_ALT,
            icon_color=theme.TEXT_PRIMARY,
            icon_size=20,
            tooltip="Start New Document",
            on_click=lambda: page.go("/"),
        ),
        ft.Icon(
            ft.Icons.CHECK_CIRCLE,
            color=theme.SUCCESS,
            size=18,
        ),
        ft.Text(
            f"{filename} — OCR Complete",
            size=14,
            weight=ft.FontWeight.W_600,
            color=theme.TEXT_PRIMARY,
            expand=True,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        ),
        ft.Container(
            content=ft.Row(
                [
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
                ],
                spacing=0,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            border=ft.Border.all(1, theme.BORDER),
            border_radius=6,
            bgcolor=theme.BG_ELEVATED,
            padding=ft.Padding.symmetric(horizontal=2),
        ),
        ft.Container(width=8),
        copy_json_button,
        download_page_markdown_button,
        download_markdown_button,
        layout_rerun_button,
        ocr_rerun_button,
        download_json_button,
    ]

    actions.bind_toolbar_controls(
        ResultsToolbarControls(
            copy_json_button=copy_json_button,
            download_json_button=download_json_button,
            download_markdown_button=download_markdown_button,
            download_page_markdown_button=download_page_markdown_button,
            layout_rerun_button=layout_rerun_button,
            ocr_rerun_button=ocr_rerun_button,
        )
    )

    toolbar = ft.Container(
        content=ft.Row(
            toolbar_controls,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
        ),
        height=48,
        padding=ft.Padding.symmetric(horizontal=12),
        bgcolor=theme.BG_SURFACE,
        border=ft.Border.only(bottom=ft.BorderSide(1, theme.BORDER)),
    )

    content_host.controls = [build_content_split_pane()]

    return ft.View(
        route=f"/results/{state.session.run_id}",
        controls=[
            ft.Column(
                [
                    build_stepper(3, page, state),
                    ft.Column([toolbar, content_host], spacing=0, expand=True)
                ],
                spacing=0,
                expand=True,
            )
        ],
        bgcolor=theme.BG_PAGE,
        padding=0,
    )
