from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, cast

import flet as ft

from free_doc_extract.ui import theme
from free_doc_extract.ui.components.bbox_editor import build_bbox_editor
from free_doc_extract.ui.state import AppState, BoundingBox, PageData


def test_resize_handle_updates_overlay_live_and_commits_on_pan_end(monkeypatch) -> None:
    monkeypatch.setattr(
        "free_doc_extract.ui.components.bbox_editor.get_image_size",
        lambda _path: (100, 100),
    )

    state = _build_selected_state()
    live_calls: list[str] = []
    commit_calls: list[str] = []

    editor = build_bbox_editor(
        state,
        lambda _box_id: None,
        lambda: commit_calls.append("commit"),
        lambda: live_calls.append("live"),
    )

    stack = _editor_stack(editor)
    box_detector = cast(ft.GestureDetector, stack.controls[1])
    top_left_handle = cast(ft.GestureDetector, stack.controls[2])
    resize_update = cast(Callable[[object], None], top_left_handle.on_pan_update)
    resize_end = cast(Callable[[object], None], top_left_handle.on_pan_end)

    resize_update(_drag_event(5, 10))

    box = state.pages[0].boxes[0]
    box_content = cast(ft.Container, box_detector.content)
    assert (box.x, box.y, box.width, box.height) == (15, 30, 25, 30)
    assert live_calls == ["live"]
    assert commit_calls == []
    assert (box_detector.left, box_detector.top) == (15, 30)
    assert (box_content.width, box_content.height) == (25, 30)
    assert (top_left_handle.left, top_left_handle.top) == (11, 26)

    resize_end(SimpleNamespace())

    assert commit_calls == ["commit"]


def test_move_handle_updates_overlay_live_and_commits_on_pan_end(monkeypatch) -> None:
    monkeypatch.setattr(
        "free_doc_extract.ui.components.bbox_editor.get_image_size",
        lambda _path: (100, 100),
    )

    state = _build_selected_state()
    live_calls: list[str] = []
    commit_calls: list[str] = []

    editor = build_bbox_editor(
        state,
        lambda _box_id: None,
        lambda: commit_calls.append("commit"),
        lambda: live_calls.append("live"),
    )

    stack = _editor_stack(editor)
    box_detector = cast(ft.GestureDetector, stack.controls[1])
    top_left_handle = cast(ft.GestureDetector, stack.controls[2])
    move_handle = cast(ft.GestureDetector, stack.controls[10])
    drag_update = cast(Callable[[object], None], move_handle.on_pan_update)
    drag_end = cast(Callable[[object], None], move_handle.on_pan_end)

    assert box_detector.on_pan_update is None

    drag_update(_drag_event(8, -4))

    box = state.pages[0].boxes[0]
    assert (box.x, box.y, box.width, box.height) == (18, 16, 30, 40)
    assert live_calls == ["live"]
    assert commit_calls == []
    assert (box_detector.left, box_detector.top) == (18, 16)
    assert (top_left_handle.left, top_left_handle.top) == (14, 12)
    assert (move_handle.left, move_handle.top) == (20, 18)

    drag_end(SimpleNamespace())

    assert commit_calls == ["commit"]


def test_overlay_colors_follow_review_kind_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        "free_doc_extract.ui.components.bbox_editor.get_image_size",
        lambda _path: (100, 100),
    )

    state = AppState()
    state.pages = [
        PageData(
            index=0,
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
                ),
            ],
        )
    ]
    state.current_page_index = 0

    editor = build_bbox_editor(
        state,
        lambda _box_id: None,
        lambda: None,
        lambda: None,
    )

    stack = _editor_stack(editor)
    text_box = cast(ft.Container, cast(ft.GestureDetector, stack.controls[1]).content)
    table_box = cast(ft.Container, cast(ft.GestureDetector, stack.controls[11]).content)
    figure_box = cast(ft.Container, cast(ft.GestureDetector, stack.controls[12]).content)
    header_box = cast(ft.Container, cast(ft.GestureDetector, stack.controls[13]).content)

    assert text_box.border == ft.Border.all(2, theme.BOX_TEXT_BLOCK)
    assert text_box.bgcolor == f"{theme.BOX_TEXT_BLOCK}1A"

    assert table_box.border == ft.Border.all(1, f"{theme.BOX_TABLE}14")
    assert table_box.bgcolor == f"{theme.BOX_TABLE}02"

    assert figure_box.border == ft.Border.all(1, f"{theme.BOX_FIGURE_IMAGE}14")
    assert figure_box.bgcolor == f"{theme.BOX_FIGURE_IMAGE}02"

    assert header_box.border == ft.Border.all(1, f"{theme.BOX_HEADER}14")
    assert header_box.bgcolor == f"{theme.BOX_HEADER}02"


def _build_selected_state() -> AppState:
    state = AppState()
    state.pages = [
        PageData(
            index=0,
            image_path="/tmp/page-0001.png",
            boxes=[
                BoundingBox(
                    id="p0-b0",
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
    state.current_page_index = 0
    state.select_box("p0-b0")
    return state


def _drag_event(dx: float, dy: float) -> SimpleNamespace:
    delta = SimpleNamespace(x=dx, y=dy)
    return SimpleNamespace(local_delta=delta, global_delta=None)


def _editor_stack(editor: ft.Container) -> ft.Stack:
    canvas = cast(ft.Column, editor.content)
    stack_row = cast(ft.Row, canvas.controls[0])
    return cast(ft.Stack, stack_row.controls[0])
