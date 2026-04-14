"""Upload drop zone with FilePicker integration."""

from __future__ import annotations

import flet as ft

from .. import theme


def build_drop_zone(file_picker: ft.FilePicker) -> ft.Container:
    def browse(e: ft.ControlEvent) -> None:
        file_picker.pick_files(
            allowed_extensions=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
            dialog_title="Select a document",
        )

    return ft.Container(
        content=ft.Column(
            [
                ft.Icon(
                    ft.Icons.UPLOAD_FILE_OUTLINED,
                    size=48,
                    color=theme.TEXT_MUTED,
                ),
                ft.Text(
                    "Drag & drop PDF or image",
                    size=14,
                    color=theme.TEXT_MUTED,
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.Container(height=8),
                ft.ElevatedButton(
                    "Browse Files",
                    on_click=browse,
                    bgcolor=theme.BG_ELEVATED,
                    color=theme.TEXT_PRIMARY,
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=6),
                    ),
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=8,
        ),
        height=280,
        border=ft.border.all(2, theme.BORDER),
        border_radius=12,
        bgcolor=theme.BG_SURFACE,
        alignment=ft.alignment.center,
    )
