from __future__ import annotations

from collections.abc import Callable

import flet as ft

from ..actions import go_to_result_route, run_workflow_action, show_error, show_layout_warning
from ..components.loading_overlay import LoadingOverlay
from ..state import AppState


def start_reviewed_ocr(
    page: ft.Page,
    state: AppState,
    loading_overlay: LoadingOverlay,
) -> None:
    if not state.session.run_id:
        return

    run_id = state.session.run_id
    run_workflow_action(
        page,
        action=lambda: state.controller.run_reviewed_ocr(run_id),
        loading=loading_overlay,
        loading_message="Running OCR...",
        error_prefix="OCR failed",
        on_success=lambda result: (
            show_layout_warning(page, state),
            go_to_result_route(page, result),
        ),
    )


def start_redetect_layout(
    page: ft.Page,
    state: AppState,
    loading_overlay: LoadingOverlay,
    rebuild: Callable[[], None],
) -> None:
    if not state.session.run_id:
        return

    input_path = state.session.current_input_path
    if not input_path:
        show_error(page, "Cannot re-detect: original input path is missing.")
        return

    has_prior_review = bool(state.session.pages)

    def start_redetect() -> None:
        run_id = state.session.run_id

        async def redetect() -> None:
            if run_id is None:
                return
            await state.controller.redetect_review(input_path, run_id)

        def on_success(_result: None) -> None:
            loading_overlay.set_active(False, "Running OCR...")
            if run_id:
                state.load_run(run_id)
            show_layout_warning(page, state)
            rebuild()

        def on_error(exc: Exception) -> None:
            loading_overlay.set_active(False, "Running OCR...")
            show_error(page, f"Re-detect failed: {exc}")

        run_workflow_action(
            page,
            action=redetect,
            loading=loading_overlay,
            loading_message="Re-detecting layout...",
            error_prefix="Re-detect failed",
            on_success=on_success,
            on_error=on_error,
        )

    if not has_prior_review:
        start_redetect()
        return

    dialog = ft.AlertDialog(modal=True)

    def close_dialog() -> None:
        dialog.open = False
        page.update()

    def confirm() -> None:
        close_dialog()
        start_redetect()

    dialog.title = ft.Text("Re-detect layout?")
    dialog.content = ft.Text(
        "Re-detecting will replace the current layout boxes on all pages. Continue?"
    )
    dialog.actions = [
        ft.TextButton("Cancel", on_click=lambda _e=None: close_dialog()),
        ft.FilledButton("Re-detect", on_click=lambda _e=None: confirm()),
    ]
    page.show_dialog(dialog)
    page.update()
