"""Main application setup: page config, theme, and routing."""

from __future__ import annotations

import flet as ft

from . import theme
from .state import AppState


def create_app(page: ft.Page) -> None:
    page.title = "Extract Document Layouts"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = theme.BG_PAGE
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=theme.PRIMARY,
            surface=theme.BG_SURFACE,
            error=theme.ERROR,
        ),
    )
    page.padding = 0

    state = AppState()
    state.load_recent_runs()

    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    def render_route() -> None:
        from .screens.results import build_results_view
        from .screens.review import build_review_view
        from .screens.upload import build_upload_view

        route = page.route or "/"
        if page.route != route:
            page.route = route

        if route == "/":
            page.title = "Step 1: Upload Document - Extract Document Layouts"
        elif route.startswith("/review/"):
            page.title = "Step 2: Review Layout - Extract Document Layouts"
        elif route.startswith("/results/"):
            page.title = "Step 3: OCR Results - Extract Document Layouts"

        page.views.clear()

        page.views.append(build_upload_view(page, state, file_picker))

        if route.startswith("/review/"):
            run_id = route.split("/review/", 1)[1]
            if state.session.run_id != run_id:
                state.load_run(run_id)
            page.views.append(build_review_view(page, state))

        elif route.startswith("/results/"):
            run_id = route.split("/results/", 1)[1]
            if state.session.run_id != run_id:
                state.load_run(run_id)
            page.views.append(build_results_view(page, state, file_picker))

        page.update()

    def route_change(_e: ft.RouteChangeEvent) -> None:
        render_route()

    def view_pop(_e: ft.ViewPopEvent) -> None:
        page.views.pop()
        if page.views:
            top = page.views[-1]
            page.go(top.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop
    render_route()

