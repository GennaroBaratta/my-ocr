"""Reusable draggable split-pane layout."""

from __future__ import annotations

import flet as ft

from .. import theme


class SplitPane(ft.Row):
    def __init__(
        self,
        left: ft.Control,
        right: ft.Control,
        initial_left_width: float = 500,
        min_left: float = 240,
        min_right: float = 240,
    ) -> None:
        self._left_width = initial_left_width
        self._min_left = min_left
        self._min_right = min_right
        self._divider_width = 6
        self._total_width: float | None = None

        self._left_container = ft.Container(
            content=left,
            width=self._clamp_left_width(initial_left_width),
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        self._right_container = ft.Container(
            content=right,
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )

        divider = ft.GestureDetector(
            content=ft.Container(
                width=self._divider_width,
                bgcolor=theme.BORDER,
                border_radius=3,
            ),
            mouse_cursor=ft.MouseCursor.RESIZE_LEFT_RIGHT,
            on_pan_update=self._on_drag,
        )

        super().__init__(
            controls=[self._left_container, divider, self._right_container],
            spacing=0,
            expand=True,
            on_size_change=self._on_size_change,
        )

    def _on_drag(self, e: ft.DragUpdateEvent) -> None:
        delta = e.local_delta or e.global_delta
        if delta is None:
            return
        self._left_width = self._clamp_left_width(self._left_width + delta.x)
        self._left_container.width = self._left_width
        try:
            self.update()
        except RuntimeError:
            pass

    def _on_size_change(self, e: ft.PageResizeEvent) -> None:
        self._total_width = e.width
        self._left_width = self._clamp_left_width(self._left_width)
        self._left_container.width = self._left_width
        try:
            self.update()
        except RuntimeError:
            pass

    def _clamp_left_width(self, width: float) -> float:
        if self._total_width is None:
            return max(self._min_left, width)

        available_width = max(0.0, self._total_width - self._divider_width)
        if available_width <= self._min_left + self._min_right:
            return max(0.0, available_width / 2)

        max_left = available_width - self._min_right
        return max(self._min_left, min(max_left, width))
