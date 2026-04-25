from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import flet as ft

from my_ocr.ui.components.split_pane import SplitPane


def test_split_pane_keeps_both_sides_above_minimum_after_resize_and_drag() -> None:
    pane = SplitPane(ft.Text("left"), ft.Text("right"))

    pane._on_size_change(SimpleNamespace(width=1000))
    pane._on_drag(_drag_event(1000))

    left_container = cast(ft.Container, pane.controls[0])
    assert left_container.width == 754

    pane._on_drag(_drag_event(-1000))

    assert left_container.width == 240


def test_split_pane_drag_keeps_existing_left_content() -> None:
    left = ft.Text("left")
    pane = SplitPane(left, ft.Text("right"))
    left_container = cast(ft.Container, pane.controls[0])

    assert left_container.content is left

    pane._on_size_change(SimpleNamespace(width=1000))
    pane._on_drag(_drag_event(100))

    assert left_container.width == 600
    assert left_container.content is left


def test_split_pane_reports_left_width_after_resize_and_drag() -> None:
    widths: list[float] = []
    pane = SplitPane(
        ft.Text("left"),
        ft.Text("right"),
        on_left_width_change=widths.append,
    )

    pane._on_size_change(SimpleNamespace(width=1000))
    pane._on_drag(_drag_event(100))

    assert widths == [500, 600]


def _drag_event(dx: float) -> SimpleNamespace:
    delta = SimpleNamespace(x=dx)
    return SimpleNamespace(local_delta=delta, global_delta=None)
