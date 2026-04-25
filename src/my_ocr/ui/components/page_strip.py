"""Left sidebar page thumbnail strip for the review workspace."""

from __future__ import annotations

from typing import Callable

import flet as ft

from .. import theme
from ..state import AppState


def build_page_strip(state: AppState, on_page_select: Callable[[int], None]) -> ft.Container:
    thumbnails: list[ft.Control] = []

    for page_data in state.session.pages:
        idx = page_data.index
        is_active = idx == state.session.current_page_index

        badge = ft.Container(
            content=ft.Text(
                str(idx + 1),
                size=10,
                weight=ft.FontWeight.W_600,
                color="white" if is_active else theme.TEXT_MUTED,
            ),
            bgcolor=theme.PRIMARY if is_active else theme.BG_ELEVATED,
            padding=ft.Padding.symmetric(horizontal=5, vertical=1),
            border_radius=3,
            right=4,
            top=4,
        )

        thumb = ft.Container(
            content=ft.Stack(
                [
                    ft.Image(
                        src=page_data.image_path,
                        width=190,
                        fit=ft.BoxFit.CONTAIN,
                        opacity=1.0 if is_active else 0.6,
                    ),
                    badge,
                ],
            ),
            border=ft.Border.all(
                2,
                theme.PRIMARY if is_active else theme.BG_SURFACE,
            ),
            border_radius=4,
            padding=2,
            on_click=lambda e, i=idx: on_page_select(i),
            ink=True,
        )
        thumbnails.append(thumb)

    return ft.Container(
        content=ft.Column(
            [
                ft.Container(
                    content=ft.Text(
                        "PAGES",
                        size=11,
                        weight=ft.FontWeight.W_600,
                        color=theme.TEXT_MUTED,
                        style=ft.TextStyle(letter_spacing=1.2),
                    ),
                    padding=ft.Padding.only(left=12, top=8, bottom=4),
                ),
                ft.Column(
                    thumbnails,
                    spacing=6,
                    scroll=ft.ScrollMode.AUTO,
                    expand=True,
                ),
            ],
            spacing=0,
            expand=True,
        ),
        width=240,
        bgcolor=theme.BG_SURFACE,
        border=ft.Border.only(right=ft.BorderSide(1, theme.BORDER)),
        padding=ft.Padding.only(bottom=8, left=8, right=8),
    )

