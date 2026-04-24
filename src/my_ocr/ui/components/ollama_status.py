"""Ollama status indicator badge."""

from __future__ import annotations

from urllib.error import URLError
from urllib.request import Request, urlopen

import flet as ft

from .. import theme


class OllamaStatus(ft.Container):
    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._dot = ft.Container(
            width=8,
            height=8,
            border_radius=4,
            bgcolor=theme.TEXT_MUTED,
        )
        self._label = ft.Text("Ollama: Checking…", size=12, color=theme.TEXT_MUTED)
        super().__init__(
            content=ft.Row(
                [self._dot, self._label],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.symmetric(horizontal=12, vertical=6),
            border=ft.Border.all(1, theme.BORDER),
            border_radius=6,
        )

    def did_mount(self) -> None:
        self.page.run_task(self._check_status)

    async def _check_status(self) -> None:
        import asyncio

        ok = await asyncio.to_thread(self._ping, self._endpoint)
        if ok:
            self._dot.bgcolor = theme.SUCCESS
            self._label.value = "Ollama: Ready"
            self._label.color = theme.TEXT_PRIMARY
        else:
            self._dot.bgcolor = theme.ERROR
            self._label.value = "Ollama: Offline"
            self._label.color = theme.ERROR
        self.update()

    @staticmethod
    def _ping(endpoint: str) -> bool:
        base = endpoint
        if "/api/" in base:
            base = base.split("/api/")[0]
        try:
            req = Request(base, method="GET")
            with urlopen(req, timeout=3) as resp:  # noqa: S310
                return resp.status == 200
        except (URLError, OSError):
            return False
