"""Results screen — split-pane with document viewer and code display."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import functools
import json
from pathlib import Path
from typing import cast

import flet as ft

from .. import theme
from ..components.code_display import build_code_display
from ..components.doc_viewer import build_doc_viewer
from ..components.split_pane import SplitPane
from ..components.stepper import build_stepper
from ..ocr_result_text import current_page_ocr_markdown_for_state, ocr_json_text_for_state
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

    content_host = ft.Row(spacing=0, expand=True)

    page_label = ft.Text(
        _page_label_text(state),
        size=13,
        color=theme.TEXT_PRIMARY,
        width=80,
        text_align=ft.TextAlign.CENTER,
    )

    def current_ocr_json_text() -> str:
        return ocr_json_text_for_state(state)

    def current_ocr_markdown_text() -> str:
        return state.ocr_markdown

    def current_page_export_markdown_text() -> str:
        return current_page_ocr_markdown_for_state(state)

    rerun_in_progress = False

    def sync_toolbar_state() -> None:
        copy_json_button.disabled = not bool(current_ocr_json_text())
        download_json_button.disabled = copy_json_button.disabled
        download_markdown_button.disabled = not bool(current_ocr_markdown_text().strip())
        download_page_markdown_button.disabled = not bool(current_page_export_markdown_text().strip())
        layout_rerun_button.disabled = rerun_in_progress
        ocr_rerun_button.disabled = rerun_in_progress

    def set_rerun_in_progress(active: bool) -> None:
        nonlocal rerun_in_progress
        rerun_in_progress = active
        sync_toolbar_state()
        page.update()

    def rebuild() -> None:
        page_label.value = _page_label_text(state)
        sync_toolbar_state()
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
        ocr_json_text = current_ocr_json_text()
        if not ocr_json_text:
            return
        await ft.Clipboard().set(ocr_json_text)
        page.show_dialog(ft.SnackBar(ft.Text("Copied OCR JSON to clipboard"), duration=1500))

    async def download_json() -> None:
        ocr_json_text = current_ocr_json_text()
        if not ocr_json_text:
            return
        save_path = await file_picker.save_file(
            file_name=f"{state.run_id or 'result'}.json",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["json"],
        )
        if save_path:
            _save_json(save_path, ocr_json_text)

    async def download_markdown() -> None:
        ocr_markdown_text = current_ocr_markdown_text()
        if not ocr_markdown_text.strip():
            return
        save_path = await file_picker.save_file(
            file_name=f"{state.run_id or 'result'}.md",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["md"],
        )
        if save_path:
            _save_markdown(save_path, ocr_markdown_text)

    async def download_page_markdown() -> None:
        page_markdown_text = current_page_export_markdown_text()
        if not page_markdown_text.strip():
            return
        current_page_number = state.current_page_number
        save_path = await file_picker.save_file(
            file_name=f"{state.run_id or 'result'}-page-{current_page_number:04d}.md",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["md"],
        )
        if save_path:
            _save_markdown(save_path, page_markdown_text)

    def reload_state(page_index: int) -> None:
        if not state.run_id:
            return
        state.load_run(state.run_id)
        if state.pages:
            state.current_page_index = min(max(page_index, 0), len(state.pages) - 1)
        else:
            state.current_page_index = 0

    def rerun_page_layout() -> None:
        from my_ocr.application.use_cases.redetect_page_layout import prepare_review_page_workflow

        if not state.run_id or rerun_in_progress:
            return
        run_id = state.run_id
        page_index = state.current_page_index
        page_number = state.current_page_number

        def on_success() -> None:
            reload_state(page_index)
            page.go(f"/review/{run_id}")

        _start_page_rerun(
            page,
            state,
            workflow=prepare_review_page_workflow,
            run_id=run_id,
            page_number=page_number,
            set_rerun_in_progress=set_rerun_in_progress,
            on_success=on_success,
            error_prefix="Page layout re-detect failed",
        )

    def rerun_page_ocr() -> None:
        from my_ocr.application.use_cases.rerun_page_ocr import run_reviewed_ocr_page_workflow

        if not state.run_id or rerun_in_progress:
            return
        run_id = state.run_id
        page_index = state.current_page_index
        page_number = state.current_page_number

        def on_success() -> None:
            reload_state(page_index)
            rebuild()

        _start_page_rerun(
            page,
            state,
            workflow=run_reviewed_ocr_page_workflow,
            run_id=run_id,
            page_number=page_number,
            set_rerun_in_progress=set_rerun_in_progress,
            on_success=on_success,
            error_prefix="Page OCR rerun failed",
        )

    copy_json_button = ft.OutlinedButton(
        "Copy OCR JSON",
        icon=ft.Icons.CONTENT_COPY,
        tooltip="Copy OCR JSON to clipboard",
        on_click=copy_clipboard,
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
        on_click=download_page_markdown,
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
        on_click=download_markdown,
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
        on_click=download_json,
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
        on_click=lambda _e=None: rerun_page_layout(),
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
        on_click=lambda _e=None: rerun_page_ocr(),
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

    sync_toolbar_state()

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


def _save_markdown(path: str, content: str) -> None:
    Path(cast(str, path)).write_text(content, encoding="utf-8")


def _save_json(path: str, content: str) -> None:
    Path(cast(str, path)).write_text(content, encoding="utf-8")


def _start_page_rerun(
    page: ft.Page,
    state: AppState,
    *,
    workflow: Callable[..., Path],
    run_id: str,
    page_number: int,
    set_rerun_in_progress: Callable[[bool], None],
    on_success: Callable[[], None],
    error_prefix: str,
) -> None:
    set_rerun_in_progress(True)

    async def do_rerun() -> None:
        try:
            await asyncio.to_thread(
                functools.partial(
                    workflow,
                    run_id,
                    page_number,
                    run_root=state.run_root,
                    layout_profile=state.layout_profile,
                )
            )
            on_success()
        except Exception as exc:
            page.show_dialog(
                ft.SnackBar(
                    ft.Text(f"{error_prefix}: {exc}"),
                    bgcolor=theme.ERROR,
                )
            )
            page.update()
        finally:
            set_rerun_in_progress(False)

    page.run_task(do_rerun)


def _page_label_text(state: AppState) -> str:
    page_count = len(state.pages)
    if page_count == 0:
        return "Page 0 / 0"
    page_number = state.current_page_number
    return f"Page {page_number} / {page_count}"
