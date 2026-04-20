from __future__ import annotations

import asyncio
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import flet as ft

from free_doc_extract.ui.components.code_display import (
    _ocr_json_text_for_state,
    _markdown_pages_for_state,
    _raw_page_text_for_state,
    build_code_display,
)
from free_doc_extract.ui.components.recent_runs import build_recent_runs
from free_doc_extract.ui.screens.review import build_review_view
from free_doc_extract.ui.screens.results import build_results_view
from free_doc_extract.ui.screens.review import _start_reviewed_ocr
from free_doc_extract.ui.screens.upload import _start_review_prep
from free_doc_extract.ui.state import AppState, BoundingBox, PageData
from tests.support import build_ocr_page, build_reviewed_layout_page, write_basic_ocr_outputs


class DummyPage:
    def __init__(self) -> None:
        self.route: str | None = None
        self.update_calls = 0
        self.dialogs: list[ft.Control] = []

    def update(self) -> None:
        self.update_calls += 1

    def go(self, route: str) -> None:
        self.route = route

    def run_task(self, task) -> None:  # noqa: ANN001
        asyncio.run(task())

    def show_dialog(self, dialog: ft.Control) -> None:
        self.dialogs.append(dialog)


def test_start_review_prep_routes_upload_into_review(tmp_path, monkeypatch) -> None:
    from free_doc_extract import workflows

    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_path = pages_dir / "page-0001.png"
    image_module.new("RGB", (100, 120), color="white").save(page_path)

    (run_dir / "reviewed_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "prepared",
                "pages": [
                    build_reviewed_layout_page(
                        page_path=str(page_path),
                        source_sdk_json_path=str(
                            run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"
                        ),
                        blocks=[],
                    )
                ],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    def fake_prepare_review_workflow(input_path: str, *, run_root: str) -> Path:
        captured["input_path"] = input_path
        captured["run_root"] = run_root
        return run_dir

    async def run_in_place(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(workflows, "prepare_review_workflow", fake_prepare_review_workflow)
    monkeypatch.setattr(asyncio, "to_thread", run_in_place)

    state = AppState()
    state.run_root = str(run_root)
    page = DummyPage()
    loading_overlay = ft.Container(visible=False)
    progress_ring = ft.ProgressRing()
    status_text = ft.Text()

    _start_review_prep(
        cast(ft.Page, page),
        state,
        "/tmp/source.pdf",
        loading_overlay,
        progress_ring,
        status_text,
    )

    assert captured == {"input_path": "/tmp/source.pdf", "run_root": str(run_root)}
    assert state.run_id == "demo-ui"
    assert page.route == "/review/demo-ui"
    assert loading_overlay.visible is False
    assert progress_ring.visible is False
    assert status_text.visible is False
    assert page.dialogs == []


def test_start_review_prep_hides_overlay_and_shows_error_on_failure(monkeypatch) -> None:
    from free_doc_extract import workflows

    def fake_prepare_review_workflow(_input_path: str, *, run_root: str) -> Path:
        raise RuntimeError(f"bad input under {run_root}")

    async def run_in_place(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(workflows, "prepare_review_workflow", fake_prepare_review_workflow)
    monkeypatch.setattr(asyncio, "to_thread", run_in_place)

    state = AppState()
    state.run_root = "/tmp/runs"
    page = DummyPage()
    loading_overlay = ft.Container(visible=False)
    progress_ring = ft.ProgressRing()
    status_text = ft.Text()

    _start_review_prep(
        cast(ft.Page, page),
        state,
        "/tmp/source.pdf",
        loading_overlay,
        progress_ring,
        status_text,
    )

    assert state.run_id is None
    assert page.route is None
    assert loading_overlay.visible is False
    assert progress_ring.visible is False
    assert status_text.visible is False
    assert len(page.dialogs) == 1

    snackbar = cast(ft.SnackBar, page.dialogs[0])
    message = cast(ft.Text, snackbar.content)
    assert message.value == "Error preparing review: bad input under /tmp/runs"


def test_start_reviewed_ocr_routes_review_into_results(tmp_path, monkeypatch) -> None:
    from free_doc_extract import workflows

    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_path = pages_dir / "page-0001.png"
    image_module.new("RGB", (100, 120), color="white").save(page_path)

    (run_dir / "reviewed_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "reviewed",
                "pages": [
                    build_reviewed_layout_page(
                        page_path=str(page_path),
                        source_sdk_json_path=str(
                            run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"
                        ),
                        blocks=[],
                    )
                ],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    def fake_run_reviewed_ocr_workflow(run: str, *, run_root: str) -> Path:
        captured["run"] = run
        captured["run_root"] = run_root
        write_basic_ocr_outputs(
            run_dir,
            markdown="# Page 1",
            json_payload={
                "pages": [
                    build_ocr_page(
                        page_path=str(page_path),
                        markdown="# Page 1",
                        sdk_json_path=str(
                            run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"
                        ),
                    )
                ]
            },
        )
        return run_dir

    monkeypatch.setattr(workflows, "run_reviewed_ocr_workflow", fake_run_reviewed_ocr_workflow)

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")

    page = DummyPage()
    loading_overlay = ft.Container(visible=False)
    progress_ring = ft.ProgressRing()
    status_text = ft.Text()

    async def run_in_place(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_in_place)

    _start_reviewed_ocr(
        cast(ft.Page, page),
        state,
        loading_overlay,
        progress_ring,
        status_text,
    )

    assert captured == {"run": "demo-ui", "run_root": str(run_root)}
    assert page.route == "/results/demo-ui"
    assert loading_overlay.visible is False
    assert progress_ring.visible is False
    assert status_text.visible is False
    assert state.ocr_markdown == "# Page 1"


def test_review_selection_updates_in_place_without_remounting_editor(monkeypatch) -> None:
    monkeypatch.setattr(
        "free_doc_extract.ui.components.bbox_editor.get_image_size",
        lambda _path: (100, 120),
    )

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

    page = DummyPage()
    view = build_review_view(cast(ft.Page, page), state)
    content_row = _review_content_row(view)
    editor = cast(ft.Container, content_row.controls[1])
    canvas = cast(ft.Column, editor.content)
    stack_row = cast(ft.Row, canvas.controls[0])
    initial_stack = cast(ft.Stack, stack_row.controls[0])
    select_box = cast(
        Callable[[object], None], cast(ft.GestureDetector, initial_stack.controls[1]).on_tap
    )

    assert len(initial_stack.controls) == 2
    select_box(SimpleNamespace())

    assert state.selected_box_id == "p0-b0"
    assert content_row.controls[1] is editor
    assert editor.content is canvas
    assert canvas.controls[0] is stack_row
    assert len(cast(ft.Stack, stack_row.controls[0]).controls) == 11
    assert cast(ft.Container, content_row.controls[2]).width == 300

    inspector = cast(ft.Container, content_row.controls[2])
    close_button = cast(
        ft.IconButton, cast(ft.Row, cast(ft.Column, inspector.content).controls[0]).controls[1]
    )
    deselect_box = cast(Callable[[], None], close_button.on_click)
    deselect_box()

    assert state.selected_box_id is None
    assert content_row.controls[1] is editor
    assert editor.content is canvas
    assert canvas.controls[0] is stack_row
    assert len(cast(ft.Stack, stack_row.controls[0]).controls) == 2
    assert cast(ft.Container, content_row.controls[2]).width == 0


def test_code_display_uses_current_page_markdown_and_raw_payload(tmp_path) -> None:
    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_one = pages_dir / "page-0001.png"
    page_two = pages_dir / "page-0002.png"
    image_module.new("RGB", (100, 120), color="white").save(page_one)
    image_module.new("RGB", (100, 120), color="white").save(page_two)

    (run_dir / "ocr.json").write_text(
        json.dumps(
            {
                "pages": [
                    build_ocr_page(
                        page_number=1,
                        page_path=str(page_one),
                        markdown="# Page 1",
                        sdk_json_path=str(
                            run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"
                        ),
                    ),
                    build_ocr_page(
                        page_number=2,
                        page_path=str(page_two),
                        markdown="# Page 2",
                        sdk_json_path=str(
                            run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json"
                        ),
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")
    state.current_page_index = 1

    assert _markdown_pages_for_state(state) == ["# Page 1", "# Page 2"]
    assert '"page_number": 2' in _raw_page_text_for_state(state)
    assert '"markdown": "# Page 2"' in _raw_page_text_for_state(state)


def test_code_display_uses_run_level_ocr_json_without_predictions(tmp_path) -> None:
    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_path = pages_dir / "page-0001.png"
    image_module.new("RGB", (100, 120), color="white").save(page_path)

    ocr_payload = {
        "pages": [
            build_ocr_page(
                page_path=str(page_path),
                markdown="# Page 1",
                sdk_json_path=str(run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"),
            )
        ],
        "summary": {"page_count": 1},
    }
    write_basic_ocr_outputs(run_dir, markdown="# Page 1", json_payload=ocr_payload)

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")

    assert state.extraction_json == {}
    assert '"page_count": 1' in _ocr_json_text_for_state(state)

    display = build_code_display(state)
    tabs = cast(ft.Tabs, display.controls[0])
    tab_content = cast(ft.Column, tabs.content)
    tab_bar = cast(ft.TabBar, tab_content.controls[0])

    assert [cast(Any, tab).label for tab in tab_bar.tabs] == ["Markdown", "OCR JSON", "Raw"]


def test_results_view_uses_ocr_json_actions_without_predictions(tmp_path) -> None:
    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_path = pages_dir / "page-0001.png"
    image_module.new("RGB", (100, 120), color="white").save(page_path)
    (run_dir / "meta.json").write_text(
        json.dumps({"input_path": "/tmp/source.pdf"}), encoding="utf-8"
    )
    write_basic_ocr_outputs(
        run_dir,
        markdown="# Page 1",
        json_payload={
            "pages": [
                build_ocr_page(
                    page_path=str(page_path),
                    markdown="# Page 1",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"),
                )
            ],
            "summary": {"page_count": 1},
        },
    )

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")

    view = build_results_view(cast(ft.Page, DummyPage()), state, ft.FilePicker())
    outer = cast(ft.Column, view.controls[0])
    inner = cast(ft.Column, outer.controls[1])
    toolbar = cast(ft.Container, inner.controls[0])
    toolbar_row = cast(ft.Row, toolbar.content)
    copy_button = cast(ft.OutlinedButton, toolbar_row.controls[7])
    download_button = cast(ft.ElevatedButton, toolbar_row.controls[8])

    assert cast(ft.Text, toolbar_row.controls[4]).value == "source.pdf — OCR Complete"
    assert cast(Any, copy_button)._values["content"] == "Copy OCR JSON"
    assert copy_button.disabled is False
    assert cast(Any, download_button)._values["content"] == "Download OCR JSON"
    assert download_button.disabled is False


def test_recent_runs_route_review_ready_and_ocr_complete_runs_correctly(tmp_path) -> None:
    run_root = tmp_path / "runs"

    review_run = run_root / "review-only"
    (review_run / "pages").mkdir(parents=True)
    (review_run / "pages" / "page-0001.png").write_bytes(b"page")
    (review_run / "ocr_raw" / "page-0001").mkdir(parents=True)
    (review_run / "ocr_raw" / "page-0001" / "page-0001_model.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (review_run / "reviewed_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "prepared",
                "pages": [
                    build_reviewed_layout_page(
                        page_path=str(review_run / "pages" / "page-0001.png"),
                        source_sdk_json_path=str(
                            review_run / "ocr_raw" / "page-0001" / "page-0001_model.json"
                        ),
                        blocks=[],
                    )
                ],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )

    ocr_run = run_root / "ocr-complete"
    write_basic_ocr_outputs(ocr_run, json_payload={"pages": [], "summary": {"page_count": 0}})

    state = AppState()
    state.run_root = str(run_root)
    state.load_recent_runs()

    page = DummyPage()
    recent_runs = build_recent_runs(cast(ft.Page, page), state)
    runs_container = cast(ft.Container, recent_runs.controls[1])
    runs_column = cast(ft.Column, runs_container.content)

    rows_by_name = {
        cast(ft.Text, cast(ft.Row, cast(ft.Container, row).content).controls[1]).value: cast(
            ft.Container, row
        )
        for row in runs_column.controls
    }

    review_row = rows_by_name["review-only"]
    review_badge = cast(ft.Container, cast(ft.Row, review_row.content).controls[3])
    assert cast(ft.Text, review_badge.content).value == "Review Ready"
    cast(Callable[[object], None], review_row.on_click)(SimpleNamespace())
    assert page.route == "/review/review-only"

    ocr_row = rows_by_name["ocr-complete"]
    ocr_badge = cast(ft.Container, cast(ft.Row, ocr_row.content).controls[3])
    assert cast(ft.Text, ocr_badge.content).value == "OCR Complete"
    cast(Callable[[object], None], ocr_row.on_click)(SimpleNamespace())
    assert page.route == "/results/ocr-complete"


def _review_content_row(view: ft.View) -> ft.Row:
    outer = cast(ft.Column, view.controls[0])
    stack = cast(ft.Stack, outer.controls[1])
    inner = cast(ft.Column, stack.controls[0])
    return cast(ft.Row, inner.controls[1])
