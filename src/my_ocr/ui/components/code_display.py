"""Markdown / OCR JSON / Raw tabbed code display for OCR results."""

from __future__ import annotations

import flet as ft

from .. import theme
from ..ocr_result_text import (
    current_page_index as _current_page_index,
    current_page_markdown_for_state as _current_page_markdown_for_state,
    current_page_ocr_markdown_for_state as _current_page_ocr_markdown_for_state,
    markdown_pages_for_state as _markdown_pages_for_state,
    ocr_json_text_for_state as _ocr_json_text_for_state,
    ocr_pages_for_state as _ocr_pages_for_state,
    raw_page_text_for_state as _raw_page_text_for_state,
)
from ..state import AppState

__all__ = [
    "_current_page_markdown_for_state",
    "_current_page_ocr_markdown_for_state",
    "_markdown_pages_for_state",
    "_ocr_json_text_for_state",
    "_ocr_pages_for_state",
    "build_code_display",
]


def build_code_display(state: AppState) -> ft.Column:
    markdown_pages = _markdown_pages_for_state(state)
    current_page_index = _current_page_index(state, markdown_pages)
    page_detail = _page_detail_text(current_page_index, markdown_pages)
    md_content = markdown_pages[current_page_index] or "_No OCR markdown available for this page._"

    ocr_json_text = _ocr_json_text_for_state(state)
    raw_text = _raw_page_text_for_state(state)

    md_view = _build_panel(
        "OCR MARKDOWN",
        page_detail,
        ft.Container(
            content=ft.Markdown(
                md_content,
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            ),
            padding=12,
        ),
    )

    if ocr_json_text:
        json_body: ft.Control = ft.Container(
            content=ft.Markdown(
                f"```json\n{ocr_json_text}\n```",
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            ),
            padding=12,
        )
    else:
        json_body = ft.Container(
            content=ft.Text(
                "No OCR JSON is available for this run yet.",
                color=theme.TEXT_MUTED,
                size=13,
            ),
            padding=12,
        )
    json_view = _build_panel(
        "OCR JSON",
        "Run-level OCR result",
        json_body,
    )

    raw_view = _build_panel(
        "OCR PAGE JSON",
        page_detail,
        ft.Container(
            content=ft.Text(
                raw_text,
                selectable=True,
                font_family="monospace",
                size=12,
                color=theme.TEXT_PRIMARY,
            ),
            padding=12,
        ),
    )

    tabs = ft.Tabs(
        selected_index=state.session.active_result_tab,
        on_change=lambda e: _on_tab_change(e, state),
        length=3,
        expand=True,
        content=ft.Column(
            [
                ft.TabBar(
                    tabs=[
                        ft.Tab(label="Markdown"),
                        ft.Tab(label="OCR JSON"),
                        ft.Tab(label="Raw"),
                    ],
                    indicator_color=theme.PRIMARY,
                    label_color=theme.TEXT_PRIMARY,
                    unselected_label_color=theme.TEXT_MUTED,
                ),
                ft.TabBarView(
                    controls=[md_view, json_view, raw_view],
                    expand=True,
                ),
            ],
            spacing=0,
            expand=True,
        ),
    )

    return ft.Column([tabs], spacing=0, expand=True)


def _on_tab_change(e: ft.Event[ft.Tabs], state: AppState) -> None:
    state.session.active_result_tab = int(e.data) if e.data else 0


def _build_panel(title: str, detail: str, body: ft.Control) -> ft.Column:
    scroll_body = ft.Column(
        [body],
        spacing=0,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )
    return ft.Column(
        [
            ft.Container(
                content=ft.Row(
                    [
                        ft.Text(
                            title,
                            size=11,
                            weight=ft.FontWeight.W_600,
                            color=theme.TEXT_MUTED,
                            style=ft.TextStyle(letter_spacing=1.2),
                        ),
                        ft.Container(expand=True),
                        ft.Text(detail, size=11, color=theme.TEXT_MUTED),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                bgcolor=theme.BG_SURFACE,
                border=ft.Border.only(bottom=ft.BorderSide(1, theme.BORDER)),
            ),
            scroll_body,
        ],
        spacing=0,
        expand=True,
    )


def _page_detail_text(current_page_index: int, markdown_pages: list[str]) -> str:
    page_count = len(markdown_pages)
    if page_count == 0:
        return "No pages"
    return f"Page {current_page_index + 1} of {page_count}"

