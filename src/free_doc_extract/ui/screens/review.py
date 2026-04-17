"""Document review workspace — 3-pane layout with page strip, bbox editor, inspector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, cast

import flet as ft

from .. import theme
from ..components.bbox_editor import build_bbox_editor, refresh_bbox_editor
from ..components.inspector import build_inspector
from ..components.page_strip import build_page_strip
from ..state import AppState


def build_review_view(page: ft.Page, state: AppState) -> ft.View:
    filename = ""
    if state.run_paths and state.run_paths.meta_path.exists():
        try:
            meta = json.loads(state.run_paths.meta_path.read_text(encoding="utf-8"))
            filename = Path(meta.get("input_path", "")).name
        except (json.JSONDecodeError, OSError):
            pass
    if not filename:
        filename = state.run_id or "Document"

    progress_bar = ft.ProgressBar(
        visible=False,
        value=None,
        color=theme.PRIMARY,
        bgcolor=theme.BG_ELEVATED,
    )

    # ── Mutable content containers ──────────────────────────────────
    content_row = ft.Row(spacing=0, expand=True)

    def rebuild() -> None:
        content_row.controls = _build_panes(
            state,
            on_page_select,
            on_box_selected,
            on_box_changed,
            on_box_live_change,
            on_deselect,
            on_remove,
        )
        page.update()

    def refresh_selection() -> None:
        if len(content_row.controls) < 3:
            rebuild()
            return

        bbox_editor = cast(ft.Container, content_row.controls[1])
        refresh_bbox_editor(
            bbox_editor,
            state,
            on_box_selected,
            on_box_changed,
            on_box_live_change,
        )
        content_row.controls[2] = build_inspector(state, on_deselect, on_box_changed, on_remove)
        page.update()

    # ── Toolbar ─────────────────────────────────────────────────────
    page_label = ft.Text(
        f"Page {state.current_page_index + 1} / {len(state.pages)}",
        size=13,
        color=theme.TEXT_PRIMARY,
        width=80,
        text_align=ft.TextAlign.CENTER,
    )
    zoom_label = ft.Text(
        f"{int(state.zoom_level * 100)}%",
        size=13,
        color=theme.TEXT_PRIMARY,
        width=44,
        text_align=ft.TextAlign.CENTER,
    )

    def prev_page() -> None:
        if state.current_page_index > 0:
            state.current_page_index -= 1
            state.select_box(None)
            page_label.value = f"Page {state.current_page_index + 1} / {len(state.pages)}"
            rebuild()

    def next_page() -> None:
        if state.current_page_index < len(state.pages) - 1:
            state.current_page_index += 1
            state.select_box(None)
            page_label.value = f"Page {state.current_page_index + 1} / {len(state.pages)}"
            rebuild()

    def zoom_in() -> None:
        state.zoom_level = min(3.0, state.zoom_level + 0.25)
        zoom_label.value = f"{int(state.zoom_level * 100)}%"
        rebuild()

    def zoom_out() -> None:
        state.zoom_level = max(0.25, state.zoom_level - 0.25)
        zoom_label.value = f"{int(state.zoom_level * 100)}%"
        rebuild()

    def run_ocr() -> None:
        state.save_reviewed_layout()
        _start_reviewed_ocr(page, state, progress_bar)

    def on_page_select(idx: int) -> None:
        state.current_page_index = idx
        state.select_box(None)
        page_label.value = f"Page {state.current_page_index + 1} / {len(state.pages)}"
        rebuild()

    def on_box_selected(box_id: str | None) -> None:
        state.select_box(box_id)
        refresh_selection()

    def on_box_changed() -> None:
        state.save_reviewed_layout()
        rebuild()

    def on_box_live_change() -> None:
        page.update()

    def on_deselect() -> None:
        state.select_box(None)
        refresh_selection()

    def on_remove(box_id: str) -> None:
        state.remove_box(box_id)
        rebuild()

    pagination_controls: list[ft.Control] = [
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
    ]

    zoom_controls: list[ft.Control] = [
        ft.IconButton(
            icon=ft.Icons.REMOVE,
            icon_size=16,
            icon_color=theme.TEXT_MUTED,
            on_click=zoom_out,
            tooltip="Zoom out",
        ),
        zoom_label,
        ft.IconButton(
            icon=ft.Icons.ADD,
            icon_size=16,
            icon_color=theme.TEXT_MUTED,
            on_click=zoom_in,
            tooltip="Zoom in",
        ),
    ]

    toolbar_controls: list[ft.Control] = [
        ft.IconButton(
            icon=ft.Icons.ARROW_BACK,
            icon_color=theme.TEXT_PRIMARY,
            icon_size=20,
            tooltip="Back",
            on_click=lambda: page.go("/"),
        ),
        ft.VerticalDivider(width=1, color=theme.BORDER),
        ft.Icon(ft.Icons.DESCRIPTION_OUTLINED, size=18, color=theme.TEXT_MUTED),
        ft.Text(
            filename,
            size=14,
            weight=ft.FontWeight.W_500,
            color=theme.TEXT_PRIMARY,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
            width=200,
        ),
        ft.Container(expand=True),
        ft.Container(
            content=ft.Row(
                pagination_controls,
                spacing=0,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            border=ft.border.all(1, theme.BORDER),
            border_radius=6,
            bgcolor=theme.BG_ELEVATED,
            padding=ft.padding.symmetric(horizontal=2),
        ),
        ft.Container(width=8),
        ft.Container(
            content=ft.Row(
                zoom_controls,
                spacing=0,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            border=ft.border.all(1, theme.BORDER),
            border_radius=6,
            bgcolor=theme.BG_ELEVATED,
            padding=ft.padding.symmetric(horizontal=2),
        ),
        ft.Container(width=8),
        ft.ElevatedButton(
            "Run OCR",
            icon=ft.Icons.PLAY_ARROW,
            on_click=run_ocr,
            tooltip="Run OCR using the reviewed layout boxes",
            bgcolor=theme.PRIMARY,
            color="white",
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=6),
            ),
        ),
    ]

    toolbar = ft.Container(
        content=ft.Row(
            toolbar_controls,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=4,
        ),
        height=48,
        padding=ft.padding.symmetric(horizontal=8),
        bgcolor=theme.BG_SURFACE,
        border=ft.border.only(bottom=ft.BorderSide(1, theme.BORDER)),
    )

    # Initial build
    content_row.controls = _build_panes(
        state,
        on_page_select,
        on_box_selected,
        on_box_changed,
        on_box_live_change,
        on_deselect,
        on_remove,
    )

    return ft.View(
        route=f"/review/{state.run_id}",
        controls=[
            ft.Column(
                [toolbar, progress_bar, content_row],
                spacing=0,
                expand=True,
            )
        ],
        bgcolor=theme.BG_PAGE,
        padding=0,
    )


def _build_panes(
    state: AppState,
    on_page_select: Callable[[int], None],
    on_box_selected: Callable[[str | None], None],
    on_box_changed: Callable[[], None],
    on_box_live_change: Callable[[], None],
    on_deselect: Callable[[], None],
    on_remove: Callable[[str], None],
) -> list[ft.Control]:
    page_strip = build_page_strip(state, on_page_select)
    bbox_editor = build_bbox_editor(state, on_box_selected, on_box_changed, on_box_live_change)
    inspector = build_inspector(state, on_deselect, on_box_changed, on_remove)

    return [page_strip, bbox_editor, inspector]


def _start_reviewed_ocr(
    page: ft.Page,
    state: AppState,
    progress_bar: ft.ProgressBar,
) -> None:
    from free_doc_extract.workflows import run_reviewed_ocr_workflow

    if not state.run_id:
        return

    progress_bar.visible = True
    page.update()

    import asyncio
    import functools

    run_id = state.run_id

    async def do_ocr() -> None:
        try:
            await asyncio.to_thread(
                functools.partial(
                    run_reviewed_ocr_workflow,
                    run_id,
                    run_root=state.run_root,
                )
            )
            progress_bar.visible = False
            state.load_run(run_id)
            page.go(f"/results/{run_id}")
        except Exception as exc:
            progress_bar.visible = False
            page.show_dialog(
                ft.SnackBar(
                    ft.Text(f"OCR failed: {exc}"),
                    bgcolor=theme.ERROR,
                )
            )
            page.update()

    page.run_task(do_ocr)
