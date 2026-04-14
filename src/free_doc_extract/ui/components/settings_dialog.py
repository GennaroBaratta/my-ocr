"""Settings dialog for configuring Ollama endpoint, model, and run root."""

from __future__ import annotations

import flet as ft

from .. import theme
from ..state import AppState


def open_settings_dialog(page: ft.Page, state: AppState) -> None:
    endpoint_field = ft.TextField(
        label="Ollama Endpoint",
        value=state.ollama_endpoint,
        border_color=theme.BORDER,
        focused_border_color=theme.PRIMARY,
        text_size=13,
    )
    model_field = ft.TextField(
        label="Ollama Model",
        value=state.ollama_model,
        border_color=theme.BORDER,
        focused_border_color=theme.PRIMARY,
        text_size=13,
    )
    run_root_field = ft.TextField(
        label="Run Root Directory",
        value=state.run_root,
        border_color=theme.BORDER,
        focused_border_color=theme.PRIMARY,
        text_size=13,
    )

    def save(e: ft.ControlEvent) -> None:
        state.ollama_endpoint = endpoint_field.value or state.ollama_endpoint
        state.ollama_model = model_field.value or state.ollama_model
        state.run_root = run_root_field.value or state.run_root
        dialog.open = False
        page.update()

    def cancel(e: ft.ControlEvent) -> None:
        dialog.open = False
        page.update()

    dialog = ft.AlertDialog(
        title=ft.Text("Settings", size=18, weight=ft.FontWeight.W_600),
        content=ft.Column(
            [endpoint_field, model_field, run_root_field],
            tight=True,
            spacing=16,
            width=400,
        ),
        actions=[
            ft.TextButton("Cancel", on_click=cancel),
            ft.ElevatedButton("Save", on_click=save, bgcolor=theme.PRIMARY, color="white"),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
        bgcolor=theme.BG_SURFACE,
    )

    page.open(dialog)
