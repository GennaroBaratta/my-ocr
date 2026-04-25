from __future__ import annotations

from collections.abc import Awaitable, Callable

import flet as ft

from my_ocr.support.filesystem import write_text

from ..actions import run_workflow_action
from ..state import AppState


def save_markdown(path: str, content: str) -> None:
    write_text(path, content)


def save_json(path: str, content: str) -> None:
    write_text(path, content)


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
