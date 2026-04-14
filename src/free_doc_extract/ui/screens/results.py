"""Results screen — split-pane with document viewer and code display."""

from __future__ import annotations

import json
from pathlib import Path

import flet as ft

from .. import theme
from ..components.code_display import build_code_display
from ..components.doc_viewer import build_doc_viewer
from ..components.split_pane import SplitPane
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

    json_str = json.dumps(state.extraction_json, indent=2, ensure_ascii=False)

    def copy_clipboard(e: ft.ControlEvent) -> None:
        page.set_clipboard(json_str)
        page.open(
            ft.SnackBar(content=ft.Text("Copied to clipboard"), duration=1500)
        )

    def download_json(e: ft.ControlEvent) -> None:
        file_picker.on_result = lambda ev: _save_json(ev, json_str)
        file_picker.save_file(
            file_name=f"{state.run_id or 'result'}.json",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["json"],
        )

    toolbar = ft.Container(
        content=ft.Row(
            [
                ft.IconButton(
                    icon=ft.Icons.ARROW_BACK,
                    icon_color=theme.TEXT_PRIMARY,
                    icon_size=20,
                    tooltip="Back",
                    on_click=lambda e: page.go(f"/review/{state.run_id}"),
                ),
                ft.Icon(
                    ft.Icons.CHECK_CIRCLE,
                    color=theme.SUCCESS,
                    size=18,
                ),
                ft.Text(
                    f"{filename} — Extraction Complete",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=theme.TEXT_PRIMARY,
                    expand=True,
                    max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                ft.OutlinedButton(
                    "Copy to Clipboard",
                    icon=ft.Icons.CONTENT_COPY,
                    on_click=copy_clipboard,
                    style=ft.ButtonStyle(
                        color=theme.TEXT_PRIMARY,
                        side=ft.BorderSide(1, theme.BORDER),
                        shape=ft.RoundedRectangleBorder(radius=6),
                    ),
                ),
                ft.ElevatedButton(
                    "Download .json",
                    icon=ft.Icons.DOWNLOAD,
                    on_click=download_json,
                    bgcolor=theme.PRIMARY,
                    color="white",
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=6),
                    ),
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
        ),
        height=48,
        padding=ft.padding.symmetric(horizontal=12),
        bgcolor=theme.BG_SURFACE,
        border=ft.border.only(bottom=ft.BorderSide(1, theme.BORDER)),
    )

    doc_viewer = build_doc_viewer(state)
    code_display = build_code_display(state)

    split = SplitPane(doc_viewer, code_display)

    return ft.View(
        route=f"/results/{state.run_id}",
        controls=[
            ft.Column(
                [toolbar, split],
                spacing=0,
                expand=True,
            )
        ],
        bgcolor=theme.BG_PAGE,
        padding=0,
    )


def _save_json(e: ft.FilePickerResultEvent, content: str) -> None:
    if e.path:
        Path(e.path).write_text(content, encoding="utf-8")
