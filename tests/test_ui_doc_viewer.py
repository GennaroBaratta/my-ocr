from __future__ import annotations

from typing import cast

import flet as ft

from my_ocr.ui import theme
from my_ocr.ui.components.doc_viewer import build_doc_viewer
from my_ocr.ui.state import AppState, BoundingBox, PageData


def test_doc_viewer_uses_review_overlay_palette_and_alpha_rules(monkeypatch) -> None:
    monkeypatch.setattr(
        "my_ocr.ui.components.doc_viewer.get_image_size",
        lambda _path: (100, 100),
    )

    state = AppState()
    state.pages = [
        PageData(
            index=0,
            page_number=1,
            image_path="/tmp/page-0001.png",
            boxes=[
                BoundingBox(
                    id="text-box",
                    page_index=0,
                    x=10,
                    y=20,
                    width=30,
                    height=40,
                    label="text",
                    selected=True,
                ),
                BoundingBox(
                    id="table-box",
                    page_index=0,
                    x=20,
                    y=30,
                    width=25,
                    height=35,
                    label="table",
                ),
                BoundingBox(
                    id="figure-box",
                    page_index=0,
                    x=30,
                    y=40,
                    width=20,
                    height=25,
                    label="image",
                ),
                BoundingBox(
                    id="header-box",
                    page_index=0,
                    x=40,
                    y=50,
                    width=15,
                    height=20,
                    label="doc_title",
                    selected=True,
                ),
            ],
        )
    ]
    state.current_page_index = 0

    viewer = build_doc_viewer(state)
    stack = _viewer_stack(viewer)

    text_box = cast(ft.Container, stack.controls[1])
    table_box = cast(ft.Container, stack.controls[2])
    figure_box = cast(ft.Container, stack.controls[3])
    header_box = cast(ft.Container, stack.controls[4])

    assert text_box.border == ft.Border.all(2, theme.BOX_TEXT_BLOCK)
    assert text_box.bgcolor == f"{theme.BOX_TEXT_BLOCK}1A"

    assert table_box.border == ft.Border.all(1, f"{theme.BOX_TABLE}14")
    assert table_box.bgcolor == f"{theme.BOX_TABLE}02"

    assert figure_box.border == ft.Border.all(1, f"{theme.BOX_FIGURE_IMAGE}0E")
    assert figure_box.bgcolor == f"{theme.BOX_FIGURE_IMAGE}01"

    assert header_box.border == ft.Border.all(2, theme.BOX_HEADER)
    assert header_box.bgcolor == f"{theme.BOX_HEADER}12"


def _viewer_stack(viewer: ft.Column) -> ft.Stack:
    canvas_container = cast(ft.Container, viewer.controls[1])
    canvas = cast(ft.Column, canvas_container.content)
    stack_row = cast(ft.Row, canvas.controls[0])
    return cast(ft.Stack, stack_row.controls[0])
