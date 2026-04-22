"""Settings dialog for configuring Ollama endpoint, model, and run root."""

from __future__ import annotations

import flet as ft

from .. import theme
from ..state import AppState


def open_settings_dialog(page: ft.Page, state: AppState) -> None:
    endpoint_field = _styled_text_field("Ollama Endpoint", state.ollama_endpoint)
    model_field = _styled_text_field("Ollama Model", state.ollama_model)
    run_root_field = _styled_text_field("Run Root Directory", state.run_root)
    
    layout_profile_field = ft.Dropdown(
        label="Layout Profile",
        value=state.layout_profile,
        options=[
            ft.dropdown.Option("auto", text="Auto"),
            ft.dropdown.Option("pp_doclayout_formula", text="PP-DocLayout Formula"),
            ft.dropdown.Option("pp_doclayout_split_formula", text="PP-DocLayout Split Formula"),
        ],
        border_color=theme.BORDER,
        focused_border_color=theme.PRIMARY,
        text_size=13,
    )

    def save() -> None:
        state.ollama_endpoint = endpoint_field.value or state.ollama_endpoint
        state.ollama_model = model_field.value or state.ollama_model
        state.run_root = run_root_field.value or state.run_root
        state.layout_profile = layout_profile_field.value or state.layout_profile
        state.load_recent_runs()
        page.pop_dialog()
        page.update()

    def cancel() -> None:
        page.pop_dialog()
        page.update()

    actions: list[ft.Control] = [
        ft.TextButton("Cancel", on_click=cancel),
        ft.Button("Save", on_click=save, bgcolor=theme.PRIMARY, color="white"),
    ]

    dialog = ft.AlertDialog(
        title=ft.Text("Settings", size=18, weight=ft.FontWeight.W_600),
        content=ft.Column(
            [endpoint_field, model_field, run_root_field, layout_profile_field],
            tight=True,
            spacing=16,
            width=400,
        ),
        actions=actions,
        actions_alignment=ft.MainAxisAlignment.END,
        bgcolor=theme.BG_SURFACE,
    )

    page.show_dialog(dialog)


def _styled_text_field(label: str, value: str) -> ft.TextField:
    field = ft.TextField()
    field.label = label
    field.value = value
    field.border_color = theme.BORDER
    field.focused_border_color = theme.PRIMARY
    field.text_size = 13
    return field
