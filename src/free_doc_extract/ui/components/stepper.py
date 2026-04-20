"""Stepper component — centered navigation bar across screens.

Completed steps are clickable so users can navigate backwards freely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from .. import theme

if TYPE_CHECKING:
    from ..state import AppState


def build_stepper(
    current_step: int,
    page: ft.Page | None = None,
    state: "AppState | None" = None,
) -> ft.Container:
    """Build a 3-step progress indicator.

    Args:
        current_step: 1 (Upload), 2 (Review Layout), 3 (OCR Results)
        page: Optional page reference - enables click-to-navigate on completed steps.
        state: Optional app state - used to build the review route with the run_id.
    """

    steps = [
        {"num": 1, "label": "Upload Document", "icon": ft.Icons.UPLOAD_FILE, "route": "/"},
        {"num": 2, "label": "Review Layout", "icon": ft.Icons.RULE, "route": None},
        {"num": 3, "label": "OCR Results", "icon": ft.Icons.DOCUMENT_SCANNER, "route": None},
    ]

    row_controls: list[ft.Control] = []

    for i, step in enumerate(steps):
        num = step["num"]
        is_active = num == current_step
        is_completed = num < current_step

        color = theme.PRIMARY if is_active else (theme.SUCCESS if is_completed else theme.TEXT_MUTED)
        bg_color = f"#1A{color[1:]}" if is_active else "transparent"
        weight = ft.FontWeight.W_600 if is_active else ft.FontWeight.NORMAL
        icon = ft.Icons.CHECK_CIRCLE if is_completed else step["icon"]

        # Resolve target route for backward navigation
        target_route: str | None = None
        if page is not None and is_completed:
            if num == 1:
                target_route = "/"
            elif num == 2 and state is not None and state.run_id:
                target_route = f"/review/{state.run_id}"

        def _make_click(route: str | None):
            if route is None:
                return None
            def handler(_e: ft.ControlEvent | None = None) -> None:
                page.go(route)  # type: ignore[union-attr]
            return handler

        on_click = _make_click(target_route)
        is_clickable = on_click is not None

        step_inner = ft.Row(
            [
                ft.Icon(icon, color=color, size=18),
                ft.Text(step["label"], color=color, weight=weight, size=14),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        step_control = ft.Container(
            content=step_inner,
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            border_radius=8,
            bgcolor=bg_color,
            on_click=on_click,
            ink=is_clickable,
            tooltip="Go back to this step" if is_clickable else None,
        )

        row_controls.append(step_control)

        if i < len(steps) - 1:
            row_controls.append(
                ft.Container(
                    width=24,
                    alignment=ft.Alignment.CENTER,
                    content=ft.Icon(
                        ft.Icons.CHEVRON_RIGHT,
                        color=theme.TEXT_MUTED,
                        size=16,
                    ),
                )
            )

    return ft.Container(
        content=ft.Stack(
            [
                # Full-width divider line at vertical center
                ft.Container(
                    height=1,
                    bgcolor=theme.BORDER,
                    top=None,
                    bottom=None,
                ),
                # Stepper row pinned to center
                ft.Row(
                    row_controls,
                    alignment=ft.MainAxisAlignment.CENTER,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=4,
                ),
            ],
            expand=True,
        ),
        height=56,
        bgcolor=theme.BG_SURFACE,
        border=ft.border.only(bottom=ft.BorderSide(1, theme.BORDER)),
    )
