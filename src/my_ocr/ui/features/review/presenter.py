"""Feature-local presenters for the layout review workspace."""

from __future__ import annotations

from collections.abc import Callable

import flet as ft

from ... import theme
from ...components.bbox_editor import build_bbox_editor
from ...components.inspector import build_inspector
from ...components.page_strip import build_page_strip
from ...state import AppState


def build_review_panes(
    state: AppState,
    on_page_select: Callable[[int], None],
    on_box_selected: Callable[[str | None], None],
    on_box_changed: Callable[[], None],
    on_box_live_change: Callable[[], None],
    on_zoom_scale_change: Callable[[float], None] | None,
    on_deselect: Callable[[], None],
    on_remove: Callable[[str], None],
) -> list[ft.Control]:
    page_strip = build_page_strip(state, on_page_select)
    bbox_editor = build_bbox_editor(
        state,
        on_box_selected,
        on_box_changed,
        on_box_live_change,
        on_zoom_scale_change,
    )
    inspector = build_inspector(state, on_deselect, on_box_changed, on_remove)

    return [page_strip, bbox_editor, inspector]


def sync_add_box_button(
    state: AppState,
    add_box_label: ft.Text,
    add_box_btn: ft.OutlinedButton,
    *,
    update: bool = False,
) -> None:
    is_adding = state.session.is_adding_box
    add_box_label.value = "Cancel Add" if is_adding else "Add Box"
    add_box_btn.icon = ft.Icons.CLOSE if is_adding else ft.Icons.ADD_BOX_OUTLINED
    add_box_btn.tooltip = "Cancel adding box" if is_adding else "Add a new layout box on this page"
    add_box_btn.style = ft.ButtonStyle(
        color=theme.TEXT_PRIMARY,
        side=ft.BorderSide(1, theme.BORDER),
        shape=ft.RoundedRectangleBorder(radius=6),
        bgcolor=f"{theme.PRIMARY}20" if is_adding else None,
    )
    if not update:
        return
    try:
        add_box_btn.update()
    except RuntimeError:
        pass
