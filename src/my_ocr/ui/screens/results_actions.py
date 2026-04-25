from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from pathlib import Path

import flet as ft

from ..actions import run_workflow_action
from ..state import AppState


@dataclass
class ResultsToolbarControls:
    copy_json_button: ft.Control
    download_json_button: ft.Control
    download_markdown_button: ft.Control
    download_page_markdown_button: ft.Control
    layout_rerun_button: ft.Control
    ocr_rerun_button: ft.Control


class ResultsScreenActions:
    def __init__(
        self,
        page: ft.Page,
        state: AppState,
        file_picker: ft.FilePicker,
        *,
        rebuild: Callable[[], None],
        current_ocr_json_text: Callable[[], str],
        current_ocr_markdown_text: Callable[[], str],
        current_page_export_markdown_text: Callable[[], str],
    ) -> None:
        self.page = page
        self.state = state
        self.file_picker = file_picker
        self.rebuild = rebuild
        self.current_ocr_json_text = current_ocr_json_text
        self.current_ocr_markdown_text = current_ocr_markdown_text
        self.current_page_export_markdown_text = current_page_export_markdown_text
        self.rerun_in_progress = False
        self.toolbar_controls: ResultsToolbarControls | None = None

    def bind_toolbar_controls(self, controls: ResultsToolbarControls) -> None:
        self.toolbar_controls = controls
        self.sync_toolbar_state()

    def sync_toolbar_state(self) -> None:
        if self.toolbar_controls is None:
            return

        controls = self.toolbar_controls
        controls.copy_json_button.disabled = not bool(self.current_ocr_json_text())
        controls.download_json_button.disabled = controls.copy_json_button.disabled
        controls.download_markdown_button.disabled = not bool(
            self.current_ocr_markdown_text().strip()
        )
        controls.download_page_markdown_button.disabled = not bool(
            self.current_page_export_markdown_text().strip()
        )
        controls.layout_rerun_button.disabled = self.rerun_in_progress
        controls.ocr_rerun_button.disabled = self.rerun_in_progress

    def prev_page(self, _e: object | None = None) -> None:
        if self.state.session.current_page_index <= 0:
            return
        self.state.session.current_page_index -= 1
        self.rebuild()

    def next_page(self, _e: object | None = None) -> None:
        if self.state.session.current_page_index >= len(self.state.session.pages) - 1:
            return
        self.state.session.current_page_index += 1
        self.rebuild()

    async def copy_clipboard(self, _e: object | None = None) -> None:
        ocr_json_text = self.current_ocr_json_text()
        if not ocr_json_text:
            return
        await ft.Clipboard().set(ocr_json_text)
        self.page.show_dialog(ft.SnackBar(ft.Text("Copied OCR JSON to clipboard"), duration=1500))

    async def download_json(self, _e: object | None = None) -> None:
        ocr_json_text = self.current_ocr_json_text()
        if not ocr_json_text:
            return
        save_path = await self.file_picker.save_file(
            file_name=f"{self.state.session.run_id or 'result'}.json",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["json"],
        )
        if save_path:
            save_json(save_path, ocr_json_text)

    async def download_markdown(self, _e: object | None = None) -> None:
        ocr_markdown_text = self.current_ocr_markdown_text()
        if not ocr_markdown_text.strip():
            return
        save_path = await self.file_picker.save_file(
            file_name=f"{self.state.session.run_id or 'result'}.md",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["md"],
        )
        if save_path:
            save_markdown(save_path, ocr_markdown_text)

    async def download_page_markdown(self, _e: object | None = None) -> None:
        page_markdown_text = self.current_page_export_markdown_text()
        if not page_markdown_text.strip():
            return
        current_page_number = self.state.current_page_number
        save_path = await self.file_picker.save_file(
            file_name=(
                f"{self.state.session.run_id or 'result'}-page-{current_page_number:04d}.md"
            ),
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["md"],
        )
        if save_path:
            save_markdown(save_path, page_markdown_text)

    def rerun_page_layout(self, _e: object | None = None) -> None:
        if not self.state.session.run_id or self.rerun_in_progress:
            return
        run_id = self.state.session.run_id
        page_index = self.state.session.current_page_index
        page_number = self.state.current_page_number

        def on_success() -> None:
            self._reload_state(page_index)
            self.page.go(f"/review/{run_id}")

        start_page_rerun(
            self.page,
            action=lambda: self.state.controller.rerun_page_layout(run_id, page_number),
            set_rerun_in_progress=self.set_rerun_in_progress,
            on_success=on_success,
            error_prefix="Page layout re-detect failed",
        )

    def rerun_page_ocr(self, _e: object | None = None) -> None:
        if not self.state.session.run_id or self.rerun_in_progress:
            return
        run_id = self.state.session.run_id
        page_index = self.state.session.current_page_index
        page_number = self.state.current_page_number

        def on_success() -> None:
            self._reload_state(page_index)
            self.rebuild()

        start_page_rerun(
            self.page,
            action=lambda: self.state.controller.rerun_page_ocr(run_id, page_number),
            set_rerun_in_progress=self.set_rerun_in_progress,
            on_success=on_success,
            error_prefix="Page OCR rerun failed",
        )

    def set_rerun_in_progress(self, active: bool) -> None:
        self.rerun_in_progress = active
        self.sync_toolbar_state()
        self.page.update()

    def _reload_state(self, page_index: int) -> None:
        if not self.state.session.run_id:
            return
        self.state.load_run(self.state.session.run_id)
        if self.state.session.pages:
            self.state.session.current_page_index = min(
                max(page_index, 0),
                len(self.state.session.pages) - 1,
            )
        else:
            self.state.session.current_page_index = 0


def save_markdown(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def save_json(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def start_page_rerun(
    page: ft.Page,
    *,
    action: Callable[[], Awaitable[object]],
    set_rerun_in_progress: Callable[[bool], None],
    on_success: Callable[[], None],
    error_prefix: str,
) -> None:
    set_rerun_in_progress(True)
    run_workflow_action(
        page,
        action=action,
        error_prefix=error_prefix,
        on_success=lambda _result: on_success(),
        on_complete=lambda: set_rerun_in_progress(False),
    )


def page_label_text(state: AppState) -> str:
    page_count = len(state.session.pages)
    if page_count == 0:
        return "Page 0 / 0"
    page_number = state.current_page_number
    return f"Page {page_number} / {page_count}"
