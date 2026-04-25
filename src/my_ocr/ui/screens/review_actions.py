from __future__ import annotations

from collections.abc import Callable

import flet as ft

from ..actions import go_to_result_route, run_workflow_action, show_error, show_layout_warning
from ..components.loading_overlay import LoadingOverlay
from ..image_utils import get_image_size
from ..state import AppState
from ..zoom import (
    ZOOM_MODE_FIT_WIDTH,
    effective_zoom_level,
    set_manual_zoom,
    toggle_fit_width_zoom,
)


def review_page_label_text(state: AppState) -> str:
    return f"Page {state.current_page_number} / {len(state.session.pages)}"


def current_zoom_scale(state: AppState) -> float:
    image_width = current_page_image_width(state)
    if image_width is None:
        return state.session.zoom_level
    return effective_zoom_level(state, image_width)


def current_page_image_width(state: AppState) -> int | None:
    page_data = state.current_page
    if not page_data:
        return None
    image_width, _image_height = get_image_size(page_data.image_path)
    return image_width


class ReviewScreenActions:
    def __init__(
        self,
        page: ft.Page,
        state: AppState,
        loading_overlay: LoadingOverlay,
        *,
        rebuild: Callable[[], None],
        refresh_selection: Callable[[], None],
        refresh_add_box_controls: Callable[[], None],
        set_page_label: Callable[[str], None],
        refresh_zoom_toolbar: Callable[[float | None], None],
        update_zoom_toolbar_controls: Callable[[], None],
    ) -> None:
        self.page = page
        self.state = state
        self.loading_overlay = loading_overlay
        self.rebuild = rebuild
        self.refresh_selection = refresh_selection
        self.refresh_add_box_controls = refresh_add_box_controls
        self.set_page_label = set_page_label
        self.refresh_zoom_toolbar = refresh_zoom_toolbar
        self.update_zoom_toolbar_controls = update_zoom_toolbar_controls

    def prev_page(self, _e: object | None = None) -> None:
        if self.state.session.current_page_index <= 0:
            return
        self.state.session.current_page_index -= 1
        self._page_changed()

    def next_page(self, _e: object | None = None) -> None:
        if self.state.session.current_page_index >= len(self.state.session.pages) - 1:
            return
        self.state.session.current_page_index += 1
        self._page_changed()

    def select_page(self, idx: int) -> None:
        self.state.session.current_page_index = idx
        self._page_changed()

    def zoom_in(self, _e: object | None = None) -> None:
        set_manual_zoom(self.state, current_zoom_scale(self.state) + 0.25)
        self.refresh_zoom_toolbar(None)
        self.rebuild()

    def zoom_out(self, _e: object | None = None) -> None:
        set_manual_zoom(self.state, current_zoom_scale(self.state) - 0.25)
        self.refresh_zoom_toolbar(None)
        self.rebuild()

    def toggle_fit_width(self, _e: object | None = None) -> None:
        toggle_fit_width_zoom(self.state, current_page_image_width(self.state))
        self.refresh_zoom_toolbar(None)
        self.rebuild()

    def run_ocr(self, _e: object | None = None) -> None:
        self.state.review_controller.save_review_layout()
        start_reviewed_ocr(self.page, self.state, self.loading_overlay)

    def toggle_add_box(self, _e: object | None = None) -> None:
        self.state.session.is_adding_box = not self.state.session.is_adding_box
        if self.state.session.is_adding_box:
            self.state.select_box(None)

        self.refresh_add_box_controls()
        self.rebuild()

    def redetect_layout(self, _e: object | None = None) -> None:
        start_redetect_layout(self.page, self.state, self.loading_overlay, self.rebuild)

    def select_box(self, box_id: str | None) -> None:
        self.state.select_box(box_id)
        self.refresh_selection()

    def box_changed(self) -> None:
        self.state.review_controller.save_review_layout()
        self.refresh_selection()

    def box_live_changed(self) -> None:
        self.page.update()

    def zoom_scale_changed(self, scale: float) -> None:
        self.refresh_zoom_toolbar(scale)
        self.update_zoom_toolbar_controls()

    def deselect_box(self) -> None:
        self.state.select_box(None)
        self.refresh_selection()

    def remove_box(self, box_id: str) -> None:
        self.state.review_controller.remove_box(box_id)
        self.refresh_selection()

    def keyboard(self, e: ft.KeyboardEvent) -> None:
        if e.key == "Delete" and self.state.session.selected_box_id:
            self.remove_box(self.state.session.selected_box_id)

    def _page_changed(self) -> None:
        self.state.select_box(None)
        self.set_page_label(review_page_label_text(self.state))
        self.rebuild()


def fit_width_icon_color(state: AppState, active_color: str, inactive_color: str) -> str:
    return active_color if state.session.zoom_mode == ZOOM_MODE_FIT_WIDTH else inactive_color


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
