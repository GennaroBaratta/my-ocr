"""Upload screen — file picker, recent runs, and inference status."""

from __future__ import annotations

from pathlib import Path

import flet as ft

from ... import theme
from ...actions import go_to_result_route, run_workflow_action, show_layout_warning
from ...components.drop_zone import build_drop_zone
from ...components.loading_overlay import LoadingOverlay, build_loading_overlay
from ...components.ollama_status import OllamaStatus
from ...components.recent_runs import build_recent_runs
from ...components.settings_dialog import open_settings_dialog
from ...components.stepper import build_stepper
from ...state import AppState


def build_upload_view(
    page: ft.Page,
    state: AppState,
    file_picker: ft.FilePicker,
) -> ft.View:
    state.load_recent_runs()

    loading_overlay = build_loading_overlay()

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
        _start_review_prep(page, state, file_path, loading_overlay)

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
    inference_badge = _build_inference_status_badge(state)

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
                                content=inference_badge,
                                left=16,
                                bottom=16,
                            ),
                            loading_overlay.control,
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
    loading_overlay: LoadingOverlay,
) -> None:
    def on_success(result: object) -> None:
        show_layout_warning(page, state)
        go_to_result_route(page, result)

    run_workflow_action(
        page,
        action=lambda: state.controller.prepare_review(file_path),
        loading=loading_overlay,
        loading_message=f"Preparing layout review for {Path(file_path).name}…",
        error_prefix="Error preparing review",
        on_success=on_success,
    )


def _build_inference_status_badge(state: AppState) -> OllamaStatus:
    inference_config = state.services.inference_config
    return OllamaStatus(
        provider=inference_config.provider,
        endpoint=inference_config.endpoint,
        endpoint_override=state.ollama_endpoint,
    )
