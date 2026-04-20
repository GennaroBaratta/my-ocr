"""Results screen — split-pane with document viewer and code display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import flet as ft

from .. import theme
from ..components.code_display import _ocr_json_text_for_state, build_code_display
from ..components.doc_viewer import build_doc_viewer
from ..components.split_pane import SplitPane
from ..components.stepper import build_stepper
from ..state import AppState


def build_results_view(
    page: ft.Page,
    state: AppState,
    file_picker: ft.FilePicker,
) -> ft.View:
    filename = ""
    if state.run_paths:
        meta_path = state.run_paths.meta_path
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                filename = Path(meta.get("input_path", "")).name
            except (json.JSONDecodeError, OSError):
                pass
        if not filename:
            filename = state.run_id or ""

    ocr_json_text = _ocr_json_text_for_state(state)
    has_ocr_json = bool(ocr_json_text)

    content_host = ft.Row(spacing=0, expand=True)

    page_label = ft.Text(
        _page_label_text(state),
        size=13,
        color=theme.TEXT_PRIMARY,
        width=80,
        text_align=ft.TextAlign.CENTER,
    )

    def rebuild() -> None:
        page_label.value = _page_label_text(state)
        content_host.controls = [SplitPane(build_doc_viewer(state), build_code_display(state))]
        page.update()

    def prev_page() -> None:
        if state.current_page_index > 0:
            state.current_page_index -= 1
            rebuild()

    def next_page() -> None:
        if state.current_page_index < len(state.pages) - 1:
            state.current_page_index += 1
            rebuild()

    async def copy_clipboard() -> None:
        await ft.Clipboard().set(ocr_json_text)
        page.show_dialog(ft.SnackBar(ft.Text("Copied OCR JSON to clipboard"), duration=1500))

    async def download_json() -> None:
        save_path = await file_picker.save_file(
            file_name=f"{state.run_id or 'result'}.json",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["json"],
        )
        if save_path:
            _save_json(save_path, ocr_json_text)

    toolbar_controls: list[ft.Control] = [
        ft.IconButton(
            icon=ft.Icons.ARROW_BACK,
            icon_color=theme.TEXT_PRIMARY,
            icon_size=20,
            tooltip="Back to Layout Review",
            on_click=lambda: page.go(f"/review/{state.run_id}"),
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
                        on_click=prev_page,
                        tooltip="Previous page",
                    ),
                    page_label,
                    ft.IconButton(
                        icon=ft.Icons.CHEVRON_RIGHT,
                        icon_size=18,
                        icon_color=theme.TEXT_MUTED,
                        on_click=next_page,
                        tooltip="Next page",
                    ),
                ],
                spacing=0,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            border=ft.border.all(1, theme.BORDER),
            border_radius=6,
            bgcolor=theme.BG_ELEVATED,
            padding=ft.padding.symmetric(horizontal=2),
        ),
        ft.Container(width=8),
        ft.OutlinedButton(
            "Copy OCR JSON",
            icon=ft.Icons.CONTENT_COPY,
            tooltip="Copy OCR JSON to clipboard",
            on_click=copy_clipboard,
            disabled=not has_ocr_json,
            style=ft.ButtonStyle(
                color=theme.TEXT_PRIMARY,
                side=ft.BorderSide(1, theme.BORDER),
                shape=ft.RoundedRectangleBorder(radius=6),
            ),
        ),
        ft.ElevatedButton(
            "Download OCR JSON",
            icon=ft.Icons.DOWNLOAD,
            tooltip="Download OCR JSON",
            on_click=download_json,
            disabled=not has_ocr_json,
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
            spacing=8,
        ),
        height=48,
        padding=ft.padding.symmetric(horizontal=12),
        bgcolor=theme.BG_SURFACE,
        border=ft.border.only(bottom=ft.BorderSide(1, theme.BORDER)),
    )

    content_host.controls = [SplitPane(build_doc_viewer(state), build_code_display(state))]

    return ft.View(
        route=f"/results/{state.run_id}",
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


def _save_json(path: str, content: str) -> None:
    Path(cast(str, path)).write_text(content, encoding="utf-8")


def _page_label_text(state: AppState) -> str:
    page_count = len(state.pages)
    if page_count == 0:
        return "Page 0 / 0"
    page_number = min(max(state.current_page_index, 0), page_count - 1) + 1
    return f"Page {page_number} / {page_count}"
