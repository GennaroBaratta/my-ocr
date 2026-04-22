"""Upload drop zone with FilePicker integration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import flet as ft

from .. import theme


def build_drop_zone(on_browse: Callable[[], Awaitable[None] | None]) -> ft.Container:
    controls: list[ft.Control] = [
        ft.Icon(
            ft.Icons.UPLOAD_FILE_OUTLINED,
            size=64,
            color=theme.PRIMARY,
        ),
        ft.Text(
            "Select a PDF or image scan",
            size=16,
            weight=ft.FontWeight.W_600,
            color=theme.TEXT_PRIMARY,
            text_align=ft.TextAlign.CENTER,
        ),
        ft.Text(
            "Supported formats: PDF, PNG, JPG, JPEG, TIF, TIFF",
            size=12,
            color=theme.TEXT_MUTED,
            text_align=ft.TextAlign.CENTER,
        ),
        ft.Container(height=8),
        ft.Button(
            "Browse Files",
            on_click=on_browse,
            bgcolor=theme.PRIMARY,
            color="white",
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=6),
            ),
        ),
    ]

    return ft.Container(
        content=ft.Column(
            controls,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=8,
        ),
        height=280,
        border=ft.Border.all(2, theme.BORDER),
        border_radius=12,
        bgcolor=theme.BG_SURFACE,
        alignment=ft.Alignment.CENTER,
    )
