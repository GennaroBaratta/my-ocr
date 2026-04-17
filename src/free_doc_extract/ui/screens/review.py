"""Document review workspace — 3-pane layout with page strip, bbox editor, inspector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import flet as ft

from .. import theme
from ..components.bbox_editor import build_bbox_editor
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
        content_row.controls = _build_panes(page, state, rebuild)
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
        _start_structured_extraction(page, state, progress_bar)

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
    content_row.controls = _build_panes(page, state, rebuild)

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
    page: ft.Page,
    state: AppState,
    rebuild: Callable[[], None],
) -> list[ft.Control]:

    def on_page_select(idx: int) -> None:
        state.current_page_index = idx
        state.select_box(None)
        rebuild()

    def on_box_selected(box_id: str | None) -> None:
        state.select_box(box_id)
        rebuild()

    def on_box_changed() -> None:
        rebuild()

    def on_deselect() -> None:
        state.select_box(None)
        rebuild()

    def on_remove(box_id: str) -> None:
        state.remove_box(box_id)
        rebuild()

    page_strip = build_page_strip(state, on_page_select)
    bbox_editor = build_bbox_editor(state, on_box_selected, on_box_changed)
    inspector = build_inspector(state, on_deselect, on_box_changed, on_remove)

    return [page_strip, bbox_editor, inspector]


def _start_structured_extraction(
    page: ft.Page,
    state: AppState,
    progress_bar: ft.ProgressBar,
) -> None:
    from free_doc_extract.workflows import run_structured_workflow

    if not state.run_id:
        return

    progress_bar.visible = True
    page.update()

    import asyncio
    import functools

    run_id = state.run_id

    async def do_extract() -> None:
        try:
            await asyncio.to_thread(
                functools.partial(
                    run_structured_workflow,
                    run_id,
                    run_root=state.run_root,
                    model=state.ollama_model,
                    endpoint=state.ollama_endpoint,
                )
            )
            progress_bar.visible = False
            state.load_run(run_id)
            page.go(f"/results/{run_id}")
        except Exception as exc:
            progress_bar.visible = False
            page.show_dialog(
                ft.SnackBar(
                    ft.Text(f"Extraction failed: {exc}"),
                    bgcolor=theme.ERROR,
                )
            )
            page.update()

    page.run_task(do_extract)
