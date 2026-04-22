"""Markdown / OCR JSON / Raw tabbed code display for OCR results."""

from __future__ import annotations

import json
from typing import Any, cast

import flet as ft

from .. import theme
from ..state import AppState


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
        selected_index=state.active_result_tab,
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
    state.active_result_tab = int(e.data) if e.data else 0


def _build_panel(title: str, detail: str, body: ft.Control) -> ft.Column:
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
            body,
        ],
        spacing=0,
        expand=True,
    )


def _page_detail_text(current_page_index: int, markdown_pages: list[str]) -> str:
    page_count = len(markdown_pages)
    if page_count == 0:
        return "No pages"
    return f"Page {current_page_index + 1} of {page_count}"


def _current_page_index(state: AppState, markdown_pages: list[str]) -> int:
    if not markdown_pages:
        return 0
    return min(max(state.current_page_index, 0), len(markdown_pages) - 1)


def _markdown_pages_for_state(state: AppState) -> list[str]:
    pages = _ocr_pages_for_state(state)
    if pages:
        markdown_pages = [
            page.get("markdown", "") if isinstance(page.get("markdown"), str) else ""
            for page in pages
        ]
        target_count = max(len(markdown_pages), len(state.pages), 1)
        while len(markdown_pages) < target_count:
            markdown_pages.append("")
        return markdown_pages
    if state.ocr_markdown.strip():
        return [state.ocr_markdown]
    return [""]


def _raw_page_text_for_state(state: AppState) -> str:
    markdown_pages = _markdown_pages_for_state(state)
    current_page_index = _current_page_index(state, markdown_pages)
    pages = _ocr_pages_for_state(state)
    if 0 <= current_page_index < len(pages):
        return json.dumps(pages[current_page_index], indent=2, ensure_ascii=False)
    return "No OCR page payload available for this page."


def _current_page_markdown_for_state(state: AppState) -> str:
    markdown_pages = _markdown_pages_for_state(state)
    current_page_index = _current_page_index(state, markdown_pages)
    if 0 <= current_page_index < len(markdown_pages):
        return markdown_pages[current_page_index]
    return ""


def _current_page_ocr_markdown_for_state(state: AppState) -> str:
    pages = _ocr_pages_for_state(state)
    if not pages:
        return ""
    current_page_index = state.current_page_index
    if not 0 <= current_page_index < len(pages):
        return ""
    markdown = pages[current_page_index].get("markdown")
    return markdown if isinstance(markdown, str) else ""


def _ocr_json_text_for_state(state: AppState) -> str:
    if not state.run_paths or not state.run_paths.ocr_json_path.exists():
        return ""
    try:
        payload = json.loads(state.run_paths.ocr_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    return json.dumps(payload, indent=2, ensure_ascii=False) if isinstance(payload, dict) else ""


def _ocr_pages_for_state(state: AppState) -> list[dict[str, Any]]:
    if not state.run_paths or not state.run_paths.ocr_json_path.exists():
        return []
    try:
        payload = json.loads(state.run_paths.ocr_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, dict):
        return []
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return []
    return [cast(dict[str, Any], page) for page in pages if isinstance(page, dict)]
