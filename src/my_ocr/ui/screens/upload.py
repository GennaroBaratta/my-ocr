"""Upload screen — file picker, recent runs, and Ollama status."""

from __future__ import annotations

from pathlib import Path

import flet as ft

from .. import theme
from ..components.drop_zone import build_drop_zone
from ..components.ollama_status import OllamaStatus
from ..components.recent_runs import build_recent_runs
from ..components.settings_dialog import open_settings_dialog
from ..components.stepper import build_stepper
from ..state import AppState


def _show_layout_warning(page: ft.Page, state: AppState) -> None:
    warning = state.layout_profile_warning()
    if not warning:
        return
    page.show_dialog(
        ft.SnackBar(
            ft.Text(warning),
            bgcolor=theme.ACCENT_YELLOW,
        )
    )


def build_upload_view(
    page: ft.Page,
    state: AppState,
    file_picker: ft.FilePicker,
) -> ft.View:
    state.load_recent_runs()

    progress_ring = ft.ProgressRing(
        visible=False,
        color=theme.PRIMARY,
        stroke_width=4,
    )
    status_text = ft.Text("", size=16, weight=ft.FontWeight.W_500, color=theme.TEXT_PRIMARY, visible=False)
    
    loading_overlay = ft.Container(
        content=ft.Column(
            [progress_ring, ft.Container(height=16), status_text],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        bgcolor=f"#CC{theme.BG_PAGE[1:]}",
        visible=False,
        alignment=ft.Alignment.CENTER,
        expand=True,
    )

    async def browse_files() -> None:
        files = await file_picker.pick_files(
            allowed_extensions=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
            dialog_title="Select a document",
        )
        if not files:
            return
        file_path = files[0].path
        if not file_path:
            return
        _start_review_prep(page, state, file_path, loading_overlay, progress_ring, status_text)

    settings_btn = ft.IconButton(
        icon=ft.Icons.SETTINGS_OUTLINED,
        icon_color=theme.TEXT_MUTED,
        icon_size=20,
        tooltip="Settings",
        on_click=lambda: open_settings_dialog(page, state),
    )

    title = ft.Text(
        "Extract Document Layouts",
        size=26,
        weight=ft.FontWeight.W_600,
        color=theme.TEXT_PRIMARY,
    )
    subtitle = ft.Text(
        "Upload a PDF or image scan to review detected layout boxes before OCR.",
        size=15,
        color=theme.TEXT_MUTED,
    )

    drop_zone = build_drop_zone(browse_files)
    recent_runs = build_recent_runs(page, state)
    ollama_badge = OllamaStatus(state.ollama_endpoint)

    content = ft.Column(
        [
            title,
            subtitle,
            ft.Container(height=20),
            drop_zone,
            ft.Container(height=20),
            recent_runs,
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=4,
    )

    return ft.View(
        route="/",
        controls=[
            ft.Column(
                [
                    build_stepper(1, page, state),
                    ft.Stack(
                        [
                            ft.Container(
                                content=ft.Container(
                                    content=content,
                                    width=600,
                                    padding=ft.Padding.symmetric(horizontal=24, vertical=16),
                                ),
                                alignment=ft.Alignment.CENTER,
                                top=0,
                                bottom=0,
                                left=0,
                                right=0,
                            ),
                            ft.Container(
                                content=settings_btn,
                                top=16,
                                right=24,
                            ),
                            ft.Container(
                                content=ollama_badge,
                                left=16,
                                bottom=16,
                            ),
                            loading_overlay,
                        ],
                        expand=True,
                    )
                ],
                spacing=0,
                expand=True,
            )
        ],
        bgcolor=theme.BG_PAGE,
        padding=0,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )


def _start_review_prep(
    page: ft.Page,
    state: AppState,
    file_path: str,
    loading_overlay: ft.Container,
    progress_ring: ft.ProgressRing,
    status_text: ft.Text,
) -> None:
    import asyncio
    import functools

    from my_ocr.application.use_cases.ocr import prepare_review_workflow

    loading_overlay.visible = True
    progress_ring.visible = True
    status_text.visible = True
    status_text.color = theme.TEXT_PRIMARY
    status_text.value = f"Preparing layout review for {Path(file_path).name}…"
    page.update()

    async def do_prepare_review() -> None:
        try:
            run_dir = await asyncio.to_thread(
                functools.partial(
                    prepare_review_workflow,
                    file_path,
                    run_root=state.run_root,
                    layout_profile=state.layout_profile,
                )
            )
            loading_overlay.visible = False
            progress_ring.visible = False
            status_text.visible = False
            state.load_run(Path(run_dir).name)
            _show_layout_warning(page, state)
            page.go(f"/review/{state.run_id}")
        except Exception as exc:
            loading_overlay.visible = False
            progress_ring.visible = False
            status_text.visible = False
            page.show_dialog(
                ft.SnackBar(
                    ft.Text(f"Error preparing review: {exc}"),
                    bgcolor=theme.ERROR,
                )
            )
            page.update()

    page.run_task(do_prepare_review)
