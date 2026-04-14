"""Upload screen — file picker, recent runs, and Ollama status."""

from __future__ import annotations

from pathlib import Path

import flet as ft

from .. import theme
from ..components.drop_zone import build_drop_zone
from ..components.ollama_status import OllamaStatus
from ..components.recent_runs import build_recent_runs
from ..components.settings_dialog import open_settings_dialog
from ..state import AppState


def build_upload_view(
    page: ft.Page,
    state: AppState,
    file_picker: ft.FilePicker,
) -> ft.View:

    progress_bar = ft.ProgressBar(
        visible=False,
        color=theme.PRIMARY,
        bgcolor=theme.BG_ELEVATED,
    )
    status_text = ft.Text("", size=12, color=theme.TEXT_MUTED, visible=False)

    def on_file_picked(e: ft.FilePickerResultEvent) -> None:
        if not e.files:
            return
        file_path = e.files[0].path
        if not file_path:
            return
        _start_pipeline(page, state, file_path, progress_bar, status_text)

    file_picker.on_result = on_file_picked

    settings_btn = ft.IconButton(
        icon=ft.Icons.SETTINGS_OUTLINED,
        icon_color=theme.TEXT_MUTED,
        icon_size=20,
        tooltip="Settings",
        on_click=lambda e: open_settings_dialog(page, state),
    )

    title = ft.Text(
        "Extract Document Layouts",
        size=26,
        weight=ft.FontWeight.W_600,
        color=theme.TEXT_PRIMARY,
    )
    subtitle = ft.Text(
        "Upload a PDF or image scan for local OCR processing.",
        size=14,
        color=theme.TEXT_MUTED,
    )

    drop_zone = build_drop_zone(file_picker)
    recent_runs = build_recent_runs(page, state)
    ollama_badge = OllamaStatus(state.ollama_endpoint)

    content = ft.Column(
        [
            ft.Row([ft.Container(expand=True), settings_btn]),
            ft.Container(height=8),
            title,
            subtitle,
            ft.Container(height=20),
            drop_zone,
            ft.Container(height=4),
            progress_bar,
            status_text,
            ft.Container(height=20),
            recent_runs,
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=4,
    )

    return ft.View(
        route="/",
        controls=[
            ft.Stack(
                [
                    ft.Container(
                        content=content,
                        width=600,
                        padding=ft.padding.symmetric(horizontal=24, vertical=16),
                        alignment=ft.alignment.top_center,
                        expand=True,
                    ),
                    ft.Container(
                        content=ollama_badge,
                        left=16,
                        bottom=16,
                    ),
                ],
                expand=True,
            )
        ],
        bgcolor=theme.BG_PAGE,
        padding=0,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )


def _start_pipeline(
    page: ft.Page,
    state: AppState,
    file_path: str,
    progress_bar: ft.ProgressBar,
    status_text: ft.Text,
) -> None:
    import asyncio
    import functools

    from free_doc_extract.workflows import run_pipeline_workflow

    progress_bar.visible = True
    progress_bar.value = None  # indeterminate
    status_text.visible = True
    status_text.value = f"Processing {Path(file_path).name}…"
    page.update()

    async def do_pipeline() -> None:
        try:
            run_dir = await asyncio.to_thread(
                functools.partial(
                    run_pipeline_workflow,
                    file_path,
                    run_root=state.run_root,
                )
            )
            progress_bar.visible = False
            status_text.visible = False
            state.load_run(Path(run_dir).name)
            page.go(f"/results/{state.run_id}")
        except Exception as exc:
            progress_bar.visible = False
            status_text.visible = True
            status_text.value = f"Error: {exc}"
            status_text.color = theme.ERROR
            page.update()

    page.run_task(do_pipeline)
