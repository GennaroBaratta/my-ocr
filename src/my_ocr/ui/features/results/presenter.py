"""Feature-local presenters for the OCR results workspace."""

from __future__ import annotations

from collections.abc import Callable

from ...components.code_display import build_code_display
from ...components.doc_viewer import build_doc_viewer, refresh_doc_viewer_available_width
from ...components.split_pane import SplitPane
from ...state import AppState


def build_results_split_pane(
    state: AppState,
    *,
    document_viewer_width: float,
    on_document_width_change: Callable[[float], None],
) -> SplitPane:
    viewer = build_doc_viewer(state, available_width=document_viewer_width)

    def on_left_width_change(width: float) -> None:
        on_document_width_change(width)
        refresh_doc_viewer_available_width(viewer, width)

    return SplitPane(
        viewer,
        build_code_display(state),
        initial_left_width=document_viewer_width,
        on_left_width_change=on_left_width_change,
    )
