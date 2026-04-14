"""Flet desktop UI for free-doc-extract."""


def main() -> None:
    import flet as ft

    from .app import create_app

    ft.app(target=create_app)
