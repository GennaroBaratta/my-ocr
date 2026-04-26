"""Feature-local toolbar primitives for the layout review workspace."""

from __future__ import annotations

import flet as ft

from ... import theme
from ...state import AppState


def build_add_box_button(
    state: AppState,
    label: ft.Text,
    on_click: ft.ControlEventHandler[ft.OutlinedButton],
) -> ft.OutlinedButton:
    return ft.OutlinedButton(
        content=label,
        icon=ft.Icons.CLOSE if state.session.is_adding_box else ft.Icons.ADD_BOX_OUTLINED,
        on_click=on_click,
        tooltip="Cancel adding box"
        if state.session.is_adding_box
        else "Add a new layout box on this page",
        style=ft.ButtonStyle(
            color=theme.TEXT_PRIMARY,
            side=ft.BorderSide(1, theme.BORDER),
            shape=ft.RoundedRectangleBorder(radius=6),
            bgcolor=f"{theme.PRIMARY}20" if state.session.is_adding_box else None,
        ),
    )
