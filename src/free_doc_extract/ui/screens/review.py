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
from ..components.stepper import build_stepper
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

    progress_ring = ft.ProgressRing(
        visible=False,
        color=theme.PRIMARY,
        stroke_width=4,
    )
    status_text = ft.Text("Running OCR...", size=16, weight=ft.FontWeight.W_500, color=theme.TEXT_PRIMARY, visible=False)
    
    loading_overlay = ft.Container(
        content=ft.Column(
            [progress_ring, ft.Container(height=16), status_text],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        bgcolor=f"#CC{theme.BG_PAGE[1:]}",
        visible=False,
        alignment=ft.Alignment.CENTER,
        expand=True,
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
        _start_reviewed_ocr(page, state, loading_overlay, progress_ring, status_text)

    def on_add_box() -> None:
        new_id = state.add_box_to_current_page()
        if new_id:
            state.select_box(new_id)
        rebuild()

    def on_redetect_layout() -> None:
        _start_redetect_layout(page, state, loading_overlay, progress_ring, status_text, rebuild)

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
        ft.IconButton(
            icon=ft.Icons.REFRESH,
            icon_color=theme.TEXT_MUTED,
            icon_size=20,
            tooltip="Re-detect layout",
            on_click=lambda _e=None: on_redetect_layout(),
        ),
        ft.OutlinedButton(
            "Add Box",
            icon=ft.Icons.ADD_BOX_OUTLINED,
            on_click=lambda _e=None: on_add_box(),
            tooltip="Add a new layout box on this page",
            style=ft.ButtonStyle(
                color=theme.TEXT_PRIMARY,
                side=ft.BorderSide(1, theme.BORDER),
                shape=ft.RoundedRectangleBorder(radius=6),
            ),
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

    def on_keyboard(e: ft.KeyboardEvent) -> None:
        if e.key == "Delete" and state.selected_box_id:
            on_remove(state.selected_box_id)

    page.on_keyboard_event = on_keyboard

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
                [
                    build_stepper(2, page, state),
                    ft.Stack(
                        [
                            ft.Column([toolbar, content_row], spacing=0, expand=True),
                            loading_overlay,
                        ],
                        expand=True,
                    ),
                ],
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
    loading_overlay: ft.Container,
    progress_ring: ft.ProgressRing,
    status_text: ft.Text,
) -> None:
    from free_doc_extract.workflows import run_reviewed_ocr_workflow

    if not state.run_id:
        return

    loading_overlay.visible = True
    progress_ring.visible = True
    status_text.visible = True
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
            loading_overlay.visible = False
            progress_ring.visible = False
            status_text.visible = False
            state.load_run(run_id)
            page.go(f"/results/{run_id}")
        except Exception as exc:
            loading_overlay.visible = False
            progress_ring.visible = False
            status_text.visible = False
            page.show_dialog(
                ft.SnackBar(
                    ft.Text(f"OCR failed: {exc}"),
                    bgcolor=theme.ERROR,
                )
            )
            page.update()

    page.run_task(do_ocr)


def _start_redetect_layout(
    page: ft.Page,
    state: AppState,
    loading_overlay: ft.Container,
    progress_ring: ft.ProgressRing,
    status_text: ft.Text,
    rebuild: Callable[[], None],
) -> None:
    from free_doc_extract.workflows import prepare_review_workflow

    if not state.run_id or not state.run_paths:
        return

    try:
        meta = json.loads(state.run_paths.meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        meta = {}
    input_path = meta.get("input_path")
    if not input_path:
        page.show_dialog(
            ft.SnackBar(
                ft.Text("Cannot re-detect: original input path is missing."),
                bgcolor=theme.ERROR,
            )
        )
        page.update()
        return

    has_prior_review = state.run_paths.reviewed_layout_path.exists()

    def start_redetect() -> None:
        import asyncio
        import functools

        run_id = state.run_id
        run_root = state.run_root
        loading_overlay.visible = True
        progress_ring.visible = True
        status_text.value = "Re-detecting layout…"
        status_text.visible = True
        page.update()

        async def do_redetect() -> None:
            try:
                await asyncio.to_thread(
                    functools.partial(
                        prepare_review_workflow,
                        input_path,
                        run=run_id,
                        run_root=run_root,
                    )
                )
                loading_overlay.visible = False
                progress_ring.visible = False
                status_text.visible = False
                status_text.value = "Running OCR..."
                if run_id:
                    state.load_run(run_id)
                rebuild()
            except Exception as exc:
                loading_overlay.visible = False
                progress_ring.visible = False
                status_text.visible = False
                status_text.value = "Running OCR..."
                page.show_dialog(
                    ft.SnackBar(
                        ft.Text(f"Re-detect failed: {exc}"),
                        bgcolor=theme.ERROR,
                    )
                )
                page.update()

        page.run_task(do_redetect)

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
