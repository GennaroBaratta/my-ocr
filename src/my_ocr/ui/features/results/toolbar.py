"""Toolbar composition for the OCR-first results workspace."""

from __future__ import annotations

import flet as ft

from ... import theme
from ...state import AppState
from .actions import ResultsScreenActions, ResultsToolbarControls


def build_results_toolbar(
    page: ft.Page,
    state: AppState,
    *,
    filename: str,
    actions: ResultsScreenActions,
    page_label: ft.Text,
) -> tuple[ft.Container, ResultsToolbarControls]:
    copy_json_button = _outlined_secondary_button(
        "Copy OCR JSON",
        ft.Icons.CONTENT_COPY,
        "Copy OCR JSON to clipboard",
        actions.copy_clipboard,
    )
    download_page_markdown_button = _secondary_button(
        "Download Page OCR Markdown",
        ft.Icons.DOWNLOAD,
        "Download OCR Markdown for this page",
        actions.download_page_markdown,
    )
    download_markdown_button = _secondary_button(
        "Download OCR Markdown",
        ft.Icons.DOWNLOAD,
        "Download OCR Markdown",
        actions.download_markdown,
    )
    download_json_button = _secondary_button(
        "Download OCR JSON",
        ft.Icons.DOWNLOAD,
        "Download OCR JSON",
        actions.download_json,
    )
    layout_rerun_button = _secondary_button(
        "Re-detect This Page Layout",
        ft.Icons.AUTO_FIX_HIGH,
        "Re-detect layout only for the active page",
        actions.rerun_page_layout,
    )
    ocr_rerun_button = _secondary_button(
        "Re-run OCR For This Page",
        ft.Icons.REFRESH,
        "Re-run OCR only for the active page",
        actions.rerun_page_ocr,
    )

    toolbar_controls = ResultsToolbarControls(
        copy_json_button=copy_json_button,
        download_json_button=download_json_button,
        download_markdown_button=download_markdown_button,
        download_page_markdown_button=download_page_markdown_button,
        layout_rerun_button=layout_rerun_button,
        ocr_rerun_button=ocr_rerun_button,
    )

    toolbar = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.IconButton(
                            icon=ft.Icons.ARROW_BACK,
                            icon_color=theme.TEXT_PRIMARY,
                            icon_size=20,
                            tooltip="Back to Layout Review",
                            on_click=lambda: page.go(f"/review/{state.session.run_id}"),
                        ),
                        ft.VerticalDivider(width=1, color=theme.BORDER),
                        ft.IconButton(
                            icon=ft.Icons.RESTART_ALT,
                            icon_color=theme.TEXT_PRIMARY,
                            icon_size=20,
                            tooltip="Start New Document",
                            on_click=lambda: page.go("/"),
                        ),
                        ft.Icon(ft.Icons.CHECK_CIRCLE, color=theme.SUCCESS, size=18),
                        ft.Column(
                            [
                                ft.Text(
                                    f"{filename} — OCR Complete",
                                    size=14,
                                    weight=ft.FontWeight.W_600,
                                    color=theme.TEXT_PRIMARY,
                                    max_lines=1,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                ),
                                ft.Text(
                                    "Review source page alongside OCR markdown and JSON",
                                    size=11,
                                    color=theme.TEXT_MUTED,
                                    max_lines=1,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                ),
                            ],
                            spacing=0,
                            expand=True,
                        ),
                        _page_navigation(actions, page_label),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=8,
                ),
                _toolbar_action_row(
                    [
                        _action_group(
                            "Export OCR",
                            [
                                copy_json_button,
                                download_page_markdown_button,
                                download_markdown_button,
                                download_json_button,
                            ],
                        ),
                        _action_group("Page tools", [layout_rerun_button, ocr_rerun_button]),
                    ]
                ),
            ],
            spacing=6,
        ),
        height=98,
        padding=ft.Padding.symmetric(horizontal=12),
        bgcolor=theme.BG_SURFACE,
        border=ft.Border.only(bottom=ft.BorderSide(1, theme.BORDER)),
    )
    return toolbar, toolbar_controls


def _page_navigation(actions: ResultsScreenActions, page_label: ft.Text) -> ft.Container:
    return ft.Container(
        content=ft.Row(
            [
                ft.Text(
                    "OCR PAGE",
                    size=10,
                    weight=ft.FontWeight.W_600,
                    color=theme.TEXT_MUTED,
                    style=ft.TextStyle(letter_spacing=1.1),
                ),
                ft.IconButton(
                    icon=ft.Icons.CHEVRON_LEFT,
                    icon_size=18,
                    icon_color=theme.TEXT_MUTED,
                    on_click=actions.prev_page,
                    tooltip="Previous page",
                ),
                page_label,
                ft.IconButton(
                    icon=ft.Icons.CHEVRON_RIGHT,
                    icon_size=18,
                    icon_color=theme.TEXT_MUTED,
                    on_click=actions.next_page,
                    tooltip="Next page",
                ),
            ],
            spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        border=ft.Border.all(1, theme.BORDER),
        border_radius=6,
        bgcolor=theme.BG_ELEVATED,
        padding=ft.Padding.symmetric(horizontal=6),
    )


def _action_group(label: str, controls: list[ft.Control]) -> ft.Container:
    return ft.Container(
        content=ft.Row(
            [
                ft.Text(
                    label.upper(),
                    size=10,
                    weight=ft.FontWeight.W_600,
                    color=theme.TEXT_MUTED,
                    style=ft.TextStyle(letter_spacing=1.1),
                ),
                *controls,
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        border=ft.Border.all(1, theme.BORDER),
        border_radius=6,
        padding=ft.Padding.symmetric(horizontal=8, vertical=4),
    )


def _toolbar_action_row(groups: list[ft.Control]) -> ft.Row:
    return ft.Row(
        groups,
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        scroll=ft.ScrollMode.AUTO,
    )


def _outlined_secondary_button(
    label: str,
    icon: ft.IconData,
    tooltip: str,
    on_click: ft.ControlEventHandler[ft.OutlinedButton],
) -> ft.OutlinedButton:
    return ft.OutlinedButton(
        label,
        icon=icon,
        tooltip=tooltip,
        on_click=on_click,
        disabled=False,
        style=ft.ButtonStyle(
            color=theme.TEXT_PRIMARY,
            side=ft.BorderSide(1, theme.BORDER),
            shape=ft.RoundedRectangleBorder(radius=6),
        ),
    )


def _secondary_button(
    label: str,
    icon: ft.IconData,
    tooltip: str,
    on_click: ft.ControlEventHandler[ft.Button],
) -> ft.Button:
    return ft.Button(
        label,
        icon=icon,
        tooltip=tooltip,
        on_click=on_click,
        disabled=False,
        bgcolor=theme.BG_ELEVATED,
        color=theme.TEXT_PRIMARY,
        style=ft.ButtonStyle(
            color=theme.TEXT_PRIMARY,
            shape=ft.RoundedRectangleBorder(radius=6),
        ),
    )
