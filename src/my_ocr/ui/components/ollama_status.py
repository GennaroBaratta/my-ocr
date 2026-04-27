"""Inference provider status indicator badge."""

from __future__ import annotations

from typing import Any, cast
from urllib.error import URLError
from urllib.request import Request, urlopen

import flet as ft

from .. import theme


class OllamaStatus(ft.Container):
    def __init__(
        self,
        *,
        provider: str,
        endpoint: str,
        endpoint_override: str | None = None,
    ) -> None:
        self._provider_label = _provider_label(provider)
        self._endpoint = endpoint_override or endpoint
        self._probe_endpoint = _probe_endpoint(self._endpoint, provider)
        self._dot = ft.Container(
            width=8,
            height=8,
            border_radius=4,
            bgcolor=theme.TEXT_MUTED,
        )
        self._label = ft.Text(
            f"{self._provider_label}: Checking…",
            size=12,
            color=theme.TEXT_MUTED,
        )
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
        page = cast(Any, self.page)
        page.run_task(self._check_status)

    async def _check_status(self) -> None:
        import asyncio

        ok = await asyncio.to_thread(self._ping, self._probe_endpoint)
        self._apply_status(ok)
        self.update()

    def _apply_status(self, ok: bool) -> None:
        if ok:
            self._dot.bgcolor = theme.SUCCESS
            self._label.value = f"{self._provider_label}: Ready"
            self._label.color = theme.TEXT_PRIMARY
        else:
            self._dot.bgcolor = theme.ERROR
            self._label.value = f"{self._provider_label}: Offline"
            self._label.color = theme.ERROR

    @staticmethod
    def _ping(endpoint: str) -> bool:
        try:
            req = Request(endpoint, method="GET")
            with urlopen(req, timeout=3) as resp:  # noqa: S310
                return resp.status == 200
        except (URLError, OSError):
            return False


def _provider_label(provider: str) -> str:
    if provider == "openai_compatible":
        return "OpenAI-compatible"
    if provider == "ollama":
        return "Ollama"
    return "Inference"


def _probe_endpoint(endpoint: str, provider: str) -> str:
    if provider == "ollama" and "/api/" in endpoint:
        return endpoint.split("/api/")[0]
    if provider == "openai_compatible" and endpoint.rstrip("/").endswith("/chat/completions"):
        return endpoint.rstrip("/")[: -len("/chat/completions")]
    return endpoint
