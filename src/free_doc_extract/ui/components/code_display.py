"""Markdown / JSON / Raw tabbed code display for extraction results."""

from __future__ import annotations

import json

import flet as ft

from .. import theme
from ..state import AppState


def build_code_display(state: AppState) -> ft.Column:
    md_content = state.ocr_markdown or "_No OCR markdown available._"
    json_str = json.dumps(state.extraction_json, indent=2, ensure_ascii=False)
    raw_text = json_str

    md_view = ft.Container(
        content=ft.Markdown(
            md_content,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        ),
        expand=True,
        padding=12,
    )

    json_view = ft.Container(
        content=ft.Markdown(
            f"```json\n{json_str}\n```",
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        ),
        expand=True,
        padding=12,
    )

    raw_view = ft.Container(
        content=ft.Text(
            raw_text,
            selectable=True,
            font_family="monospace",
            size=12,
            color=theme.TEXT_PRIMARY,
        ),
        expand=True,
        padding=12,
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
                        ft.Tab(label="JSON"),
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
