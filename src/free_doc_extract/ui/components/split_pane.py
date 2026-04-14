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
        min_left: float = 200,
        max_left: float = 900,
    ) -> None:
        self._left_width = initial_left_width
        self._min_left = min_left
        self._max_left = max_left

        self._left_container = ft.Container(
            content=left,
            width=initial_left_width,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        self._right_container = ft.Container(
            content=right,
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )

        divider = ft.GestureDetector(
            content=ft.Container(
                width=6,
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
        )

    def _on_drag(self, e: ft.DragUpdateEvent) -> None:
        self._left_width = max(
            self._min_left, min(self._max_left, self._left_width + e.delta_x)
        )
        self._left_container.width = self._left_width
        self.update()
