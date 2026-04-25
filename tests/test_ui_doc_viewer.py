from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, cast

import flet as ft

from my_ocr.ui import theme
from my_ocr.ui.components.doc_viewer import build_doc_viewer, refresh_doc_viewer_available_width
from my_ocr.ui.components.split_pane import SplitPane
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


def test_doc_viewer_defaults_to_fit_width_and_exposes_toolbar_button(monkeypatch) -> None:
    monkeypatch.setattr(
        "my_ocr.ui.components.doc_viewer.get_image_size",
        lambda _path: (100, 200),
    )

    state = AppState()
    state.zoom_fit_width = 82
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
                )
            ],
        )
    ]

    viewer = build_doc_viewer(state)
    header_row = cast(ft.Row, cast(ft.Container, viewer.controls[0]).content)
    fit_width_button = cast(ft.IconButton, header_row.controls[2])
    zoom_label = cast(ft.Text, header_row.controls[4])
    stack = _viewer_stack(viewer)
    text_box = cast(ft.Container, stack.controls[1])

    assert state.zoom_mode == "fit_width"
    assert fit_width_button.icon == ft.Icons.WIDTH_FULL
    assert fit_width_button.icon_color == theme.PRIMARY
    assert zoom_label.value == "Fit 50%"
    assert stack.width == 50
    assert stack.height == 100
    assert text_box.left == 5
    assert text_box.width == 15


def test_doc_viewer_resize_refreshes_fit_width_canvas_and_header(monkeypatch) -> None:
    monkeypatch.setattr(
        "my_ocr.ui.components.doc_viewer.get_image_size",
        lambda _path: (100, 200),
    )

    state = AppState()
    state.zoom_fit_width = 82
    state.pages = [PageData(index=0, page_number=1, image_path="/tmp/page-0001.png")]

    viewer = build_doc_viewer(state)
    header_row = cast(ft.Row, cast(ft.Container, viewer.controls[0]).content)
    zoom_label = cast(ft.Text, header_row.controls[4])
    canvas_container = cast(ft.Container, viewer.controls[1])
    on_size_change = cast(Callable[[object], None], canvas_container.on_size_change)

    on_size_change(SimpleNamespace(width=132, control=canvas_container))

    stack = _viewer_stack(viewer)
    assert state.zoom_fit_width == 132
    assert zoom_label.value == "Fit 100%"
    assert stack.width == 100
    assert stack.height == 200


def test_doc_viewer_fit_width_button_toggles_to_manual_without_page_jump(monkeypatch) -> None:
    monkeypatch.setattr(
        "my_ocr.ui.components.doc_viewer.get_image_size",
        lambda _path: (100, 200),
    )

    state = AppState()
    state.zoom_fit_width = 132
    state.pages = [PageData(index=0, page_number=1, image_path="/tmp/page-0001.png")]

    viewer = build_doc_viewer(state)
    header_row = cast(ft.Row, cast(ft.Container, viewer.controls[0]).content)
    fit_width_button = cast(ft.IconButton, header_row.controls[2])
    fit_width_button.on_click(SimpleNamespace(page=SimpleNamespace(update=lambda: None)))

    zoom_label = cast(ft.Text, header_row.controls[4])
    stack = _viewer_stack(viewer)

    assert state.zoom_mode == "manual"
    assert state.zoom_level == 1.0
    assert fit_width_button.icon_color == theme.TEXT_MUTED
    assert zoom_label.value == "100%"
    assert stack.width == 100
    assert stack.height == 200


def test_doc_viewer_split_pane_width_growth_refreshes_fit_width_canvas(monkeypatch) -> None:
    monkeypatch.setattr(
        "my_ocr.ui.components.doc_viewer.get_image_size",
        lambda _path: (100, 200),
    )

    state = AppState()
    state.pages = [PageData(index=0, page_number=1, image_path="/tmp/page-0001.png")]
    viewer = build_doc_viewer(state, available_width=82)
    pane = SplitPane(
        viewer,
        ft.Text("right"),
        initial_left_width=82,
        min_left=50,
        min_right=50,
        on_left_width_change=lambda width: refresh_doc_viewer_available_width(viewer, width),
    )

    pane._on_size_change(SimpleNamespace(width=300))
    pane._on_drag(_drag_event(50))

    stack = _viewer_stack(viewer)
    assert state.zoom_fit_width == 132
    assert stack.width == 100
    assert stack.height == 200


def test_doc_viewer_embeds_local_image_bytes_for_web_client(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "my_ocr.ui.components.doc_viewer.get_image_size",
        lambda _path: (100, 200),
    )
    image_path = tmp_path / "page-0001.png"
    image_path.write_bytes(b"local image bytes")

    state = AppState()
    state.pages = [PageData(index=0, page_number=1, image_path=str(image_path))]

    viewer = build_doc_viewer(state)
    stack = _viewer_stack(viewer)
    image = cast(ft.Image, stack.controls[0])

    assert image.src == b"local image bytes"


def test_doc_viewer_manual_zoom_switches_out_of_fit_width(monkeypatch) -> None:
    monkeypatch.setattr(
        "my_ocr.ui.components.doc_viewer.get_image_size",
        lambda _path: (100, 200),
    )

    state = AppState()
    state.zoom_fit_width = 132
    state.pages = [PageData(index=0, page_number=1, image_path="/tmp/page-0001.png")]
    viewer = build_doc_viewer(state)
    header_row = cast(ft.Row, cast(ft.Container, viewer.controls[0]).content)
    zoom_in_button = cast(ft.IconButton, header_row.controls[5])
    zoom_in_button.on_click(SimpleNamespace(page=SimpleNamespace(update=lambda: None)))

    zoom_label = cast(ft.Text, header_row.controls[4])
    fit_width_button = cast(ft.IconButton, header_row.controls[2])
    stack = _viewer_stack(viewer)

    assert state.zoom_mode == "manual"
    assert state.zoom_level == 1.25
    assert zoom_label.value == "125%"
    assert fit_width_button.icon_color == theme.TEXT_MUTED
    assert stack.width == 125


def _viewer_stack(viewer: ft.Column) -> ft.Stack:
    canvas_container = cast(ft.Container, viewer.controls[1])
    canvas = cast(ft.Column, canvas_container.content)
    stack_row = cast(ft.Row, canvas.controls[0])
    return cast(ft.Stack, stack_row.controls[0])


def _drag_event(dx: float) -> SimpleNamespace:
    delta = SimpleNamespace(x=dx)
    return SimpleNamespace(local_delta=delta, global_delta=None)
