"""Reusable loading overlay for workflow screens."""

from __future__ import annotations

from dataclasses import dataclass

import flet as ft

from .. import theme


@dataclass
class LoadingOverlay:
    control: ft.Container
    progress_ring: ft.ProgressRing
    status_text: ft.Text

    def set_active(self, active: bool, message: str | None = None) -> None:
        if message is not None:
            self.status_text.value = message
        self.control.visible = active
        self.progress_ring.visible = active
        self.status_text.visible = active


def build_loading_overlay(message: str = "") -> LoadingOverlay:
    progress_ring = ft.ProgressRing(
        visible=False,
        color=theme.PRIMARY,
        stroke_width=4,
    )
    status_text = ft.Text(
        message,
        size=16,
        weight=ft.FontWeight.W_500,
        color=theme.TEXT_PRIMARY,
        visible=False,
    )
    control = ft.Container(
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
    return LoadingOverlay(control, progress_ring, status_text)
