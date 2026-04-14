"""Right-side properties inspector panel for the selected bounding box."""

from __future__ import annotations

from typing import Callable

import flet as ft

from .. import theme
from ..state import AppState, BoundingBox


BLOCK_TYPES = ["Text Block", "Table", "Figure/Image", "Header"]

LABEL_TO_BLOCK_TYPE = {
    "doc_title": "Header",
    "paragraph_title": "Header",
    "table": "Table",
    "figure": "Figure/Image",
    "image": "Figure/Image",
}


def build_inspector(
    state: AppState,
    on_deselect: Callable[[], None],
    on_change: Callable[[], None],
    on_remove: Callable[[str], None],
) -> ft.Container:
    box = state.get_selected_box()
    if not box:
        return ft.Container(width=0)

    block_type_value = LABEL_TO_BLOCK_TYPE.get(box.label, "Text Block")

    def on_type_change(e: ft.ControlEvent) -> None:
        reverse = {v: k for k, v in LABEL_TO_BLOCK_TYPE.items()}
        box.label = reverse.get(e.data, box.label)
        on_change()

    def make_coord_handler(attr: str):
        def handler(e: ft.ControlEvent) -> None:
            try:
                setattr(box, attr, float(e.data))
            except (ValueError, TypeError):
                pass
            on_change()
        return handler

    confidence_pct = int(box.confidence * 100)

    header = ft.Row(
        [
            ft.Text(
                "PROPERTIES",
                size=11,
                weight=ft.FontWeight.W_600,
                color=theme.TEXT_MUTED,
                letter_spacing=1.2,
                expand=True,
            ),
            ft.IconButton(
                icon=ft.Icons.CLOSE,
                icon_size=16,
                icon_color=theme.TEXT_MUTED,
                tooltip="Close",
                on_click=lambda e: on_deselect(),
            ),
        ],
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    info_row = ft.Row(
        [
            ft.Container(
                content=ft.Icon(
                    _icon_for_type(block_type_value),
                    size=20,
                    color=theme.PRIMARY,
                ),
                bgcolor=f"{theme.PRIMARY}33",
                border_radius=6,
                padding=6,
            ),
            ft.Column(
                [
                    ft.Text(
                        block_type_value,
                        size=14,
                        weight=ft.FontWeight.W_600,
                        color=theme.TEXT_PRIMARY,
                    ),
                    ft.Text(
                        box.id,
                        size=11,
                        color=theme.TEXT_MUTED,
                        font_family="monospace",
                    ),
                ],
                spacing=2,
            ),
        ],
        spacing=10,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    type_dropdown = ft.Dropdown(
        label="Block Type",
        value=block_type_value,
        options=[ft.dropdown.Option(t) for t in BLOCK_TYPES],
        border_color=theme.BORDER,
        focused_border_color=theme.PRIMARY,
        text_size=13,
        on_change=on_type_change,
    )

    confidence_row = ft.Column(
        [
            ft.Row(
                [
                    ft.Text("Confidence", size=11, color=theme.TEXT_MUTED),
                    ft.Container(expand=True),
                    ft.Text(
                        f"{confidence_pct}%",
                        size=12,
                        weight=ft.FontWeight.W_500,
                        color=theme.TEXT_PRIMARY,
                    ),
                ],
            ),
            ft.ProgressBar(
                value=box.confidence,
                color=theme.SUCCESS,
                bgcolor=theme.BG_ELEVATED,
                bar_height=4,
                border_radius=2,
            ),
        ],
        spacing=4,
    )

    coord_fields = ft.Column(
        [
            ft.Text(
                "COORDINATES (PX)",
                size=10,
                weight=ft.FontWeight.W_600,
                color=theme.TEXT_MUTED,
                letter_spacing=1,
            ),
            ft.Row(
                [
                    _coord_field("X (Left)", box.x, make_coord_handler("x")),
                    _coord_field("Y (Top)", box.y, make_coord_handler("y")),
                ],
                spacing=8,
            ),
            ft.Row(
                [
                    _coord_field("Width", box.width, make_coord_handler("width")),
                    _coord_field("Height", box.height, make_coord_handler("height")),
                ],
                spacing=8,
            ),
        ],
        spacing=8,
    )

    remove_btn = ft.OutlinedButton(
        "Remove",
        icon=ft.Icons.DELETE_OUTLINE,
        icon_color=theme.ERROR,
        on_click=lambda e: on_remove(box.id),
        style=ft.ButtonStyle(
            color=theme.TEXT_PRIMARY,
            side=ft.BorderSide(1, theme.BORDER),
            shape=ft.RoundedRectangleBorder(radius=6),
        ),
        width=float("inf"),
    )

    return ft.Container(
        content=ft.Column(
            [
                header,
                ft.Divider(height=1, color=theme.BORDER),
                info_row,
                ft.Container(height=8),
                type_dropdown,
                ft.Container(height=12),
                confidence_row,
                ft.Container(height=12),
                coord_fields,
                ft.Container(height=16),
                remove_btn,
            ],
            spacing=4,
            scroll=ft.ScrollMode.AUTO,
        ),
        width=300,
        bgcolor=theme.BG_SURFACE,
        border=ft.border.only(left=ft.BorderSide(1, theme.BORDER)),
        padding=ft.padding.all(12),
    )


def _coord_field(
    label: str,
    value: float,
    on_change: Callable[[ft.ControlEvent], None],
) -> ft.TextField:
    return ft.TextField(
        label=label,
        value=str(int(value)),
        keyboard_type=ft.KeyboardType.NUMBER,
        text_size=12,
        label_style=ft.TextStyle(size=10, color=theme.TEXT_MUTED),
        border_color=theme.BORDER,
        focused_border_color=theme.PRIMARY,
        on_change=on_change,
        expand=True,
        content_padding=ft.padding.symmetric(horizontal=8, vertical=8),
    )


def _icon_for_type(block_type: str) -> str:
    return {
        "Header": ft.Icons.TITLE,
        "Table": ft.Icons.TABLE_CHART,
        "Figure/Image": ft.Icons.IMAGE,
        "Text Block": ft.Icons.NOTES,
    }.get(block_type, ft.Icons.NOTES)
