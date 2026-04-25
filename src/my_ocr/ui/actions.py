"""Shared Flet workflow action helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

import flet as ft

from . import theme
from .components.loading_overlay import LoadingOverlay
from .state import AppState

T = TypeVar("T")


def show_error(page: ft.Page, message: str) -> None:
    page.show_dialog(
        ft.SnackBar(
            ft.Text(message),
            bgcolor=theme.ERROR,
        )
    )
    page.update()


def show_layout_warning(page: ft.Page, state: AppState) -> None:
    warning = state.layout_profile_warning()
    if not warning:
        return
    page.show_dialog(
        ft.SnackBar(
            ft.Text(warning),
            bgcolor=theme.ACCENT_YELLOW,
        )
    )


def go_to_result_route(page: ft.Page, result: object) -> None:
    route = getattr(result, "route", None)
    if route:
        page.go(route)


def run_workflow_action(
    page: ft.Page,
    *,
    action: Callable[[], Awaitable[T]],
    error_prefix: str,
    loading: LoadingOverlay | None = None,
    loading_message: str | None = None,
    on_success: Callable[[T], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
    on_complete: Callable[[], None] | None = None,
) -> None:
    if loading is not None:
        loading.set_active(True, loading_message)
        page.update()

    async def run() -> None:
        try:
            result = await action()
            if loading is not None:
                loading.set_active(False)
            if on_success is not None:
                on_success(result)
        except Exception as exc:
            if loading is not None:
                loading.set_active(False)
            if on_error is not None:
                on_error(exc)
            else:
                show_error(page, f"{error_prefix}: {exc}")
        finally:
            if on_complete is not None:
                on_complete()

    page.run_task(run)
