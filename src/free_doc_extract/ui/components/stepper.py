"""Top breadcrumb showing the current step of the 3-step pipeline."""

from __future__ import annotations

import flet as ft

from .. import theme
from ..state import AppState


_STEPS: tuple[tuple[str, str], ...] = (
    ("Upload", "/"),
    ("Review", "review"),
    ("Results", "results"),
)


def build_stepper(current: int, page: ft.Page, state: AppState) -> ft.Container:
    items: list[ft.Control] = []
    for idx, (label, key) in enumerate(_STEPS, start=1):
        items.append(_step_chip(idx, label, key, current, page, state))
        if idx < len(_STEPS):
            items.append(
                ft.Container(
                    width=24,
                    height=1,
                    bgcolor=theme.BORDER,
                    margin=ft.margin.symmetric(horizontal=4),
                )
            )

    return ft.Container(
        content=ft.Row(
            items,
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=0,
        ),
        height=40,
        bgcolor=theme.BG_SURFACE,
        border=ft.border.only(bottom=ft.BorderSide(1, theme.BORDER)),
        padding=ft.padding.symmetric(horizontal=12),
    )


def _step_chip(
    index: int,
    label: str,
    key: str,
    current: int,
    page: ft.Page,
    state: AppState,
) -> ft.Control:
    is_current = index == current
    is_past = index < current
    color = theme.PRIMARY if is_current else (theme.TEXT_PRIMARY if is_past else theme.TEXT_MUTED)
    circle_bg = theme.PRIMARY if is_current else (theme.SUCCESS if is_past else theme.BG_ELEVATED)
    circle_fg = "white" if (is_current or is_past) else theme.TEXT_MUTED

    circle = ft.Container(
        width=22,
        height=22,
        border_radius=11,
        bgcolor=circle_bg,
        alignment=ft.Alignment.CENTER,
        content=ft.Text(str(index), size=12, color=circle_fg, weight=ft.FontWeight.W_600),
    )
    text = ft.Text(
        label,
        size=13,
        color=color,
        weight=ft.FontWeight.W_600 if is_current else ft.FontWeight.W_500,
    )

    chip = ft.Container(
        content=ft.Row(
            [circle, text],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.padding.symmetric(horizontal=10, vertical=4),
        border_radius=6,
    )

    target = _step_route(key, state)
    if target and not is_current:
        chip.ink = True
        chip.on_click = lambda _e=None, route=target: page.go(route)

    return chip


def _step_route(key: str, state: AppState) -> str | None:
    if key == "/":
        return "/"
    if not state.run_id:
        return None
    if key == "review":
        return f"/review/{state.run_id}"
    if key == "results":
        return f"/results/{state.run_id}"
    return None
