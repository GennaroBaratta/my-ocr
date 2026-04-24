"""Flet desktop UI for my-ocr."""

from __future__ import annotations


def main(*, web: bool = False, host: str | None = None, port: int = 0) -> None:
    import flet as ft

    from .app import create_app

    if web:
        ft.run(
            create_app,
            host=host,
            port=port,
            view=ft.AppView.WEB_BROWSER,
            route_url_strategy=ft.RouteUrlStrategy.PATH,
        )
        return

    ft.app(target=create_app)
