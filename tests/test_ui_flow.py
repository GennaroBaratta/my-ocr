from __future__ import annotations

import asyncio
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import flet as ft

from my_ocr.ui.components.code_display import (
    _ocr_json_text_for_state,
    _markdown_pages_for_state,
    _raw_page_text_for_state,
    build_code_display,
)
from my_ocr.ui.components.recent_runs import build_recent_runs
from my_ocr.ui.screens.review import build_review_view
from my_ocr.ui.screens.results import build_results_view
from my_ocr.ui.screens.review import _start_reviewed_ocr
from my_ocr.ui.screens.upload import _start_review_prep
from my_ocr.ui.state import AppState, BoundingBox, PageData
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


def _toolbar_buttons_by_content(toolbar_row: ft.Row) -> dict[str, ft.Control]:
    buttons: dict[str, ft.Control] = {}
    for control in toolbar_row.controls:
        values = getattr(control, "_values", None)
        content = values.get("content") if isinstance(values, dict) else None
        if isinstance(content, str):
            buttons[content] = control
    return buttons


def test_start_review_prep_routes_upload_into_review(tmp_path, monkeypatch) -> None:
    from my_ocr import workflows

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

    def fake_prepare_review_workflow(input_path: str, *, run_root: str, **kwargs) -> Path:
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
    from my_ocr import workflows

    def fake_prepare_review_workflow(_input_path: str, *, run_root: str, **kwargs) -> Path:
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


def test_start_review_prep_shows_layout_profile_warning_from_metadata(tmp_path, monkeypatch) -> None:
    from my_ocr import workflows

    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_path = pages_dir / "page-0001.png"
    image_module.new("RGB", (100, 120), color="white").save(page_path)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "input_path": "/tmp/source.pdf",
                "layout_diagnostics": {
                    "layout_profile_warning": "Proceeding with existing config/default mappings."
                },
            }
        ),
        encoding="utf-8",
    )
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

    async def run_in_place(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(workflows, "prepare_review_workflow", lambda *_args, **_kwargs: run_dir)
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

    assert len(page.dialogs) == 1
    snackbar = cast(ft.SnackBar, page.dialogs[0])
    message = cast(ft.Text, snackbar.content)
    assert message.value == "Proceeding with existing config/default mappings."


def test_start_reviewed_ocr_routes_review_into_results(tmp_path, monkeypatch) -> None:
    from my_ocr import workflows

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

    def fake_run_reviewed_ocr_workflow(run: str, *, run_root: str, **kwargs) -> Path:
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


def test_start_reviewed_ocr_shows_layout_profile_warning_from_metadata(
    tmp_path, monkeypatch
) -> None:
    from my_ocr import workflows

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
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "input_path": "/tmp/source.pdf",
                "layout_diagnostics": {
                    "layout_profile_warning": "Proceeding with existing config/default mappings."
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run_reviewed_ocr_workflow(_run: str, *, run_root: str, **kwargs) -> Path:
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
        (run_dir / "meta.json").write_text(
            json.dumps(
                {
                    "input_path": "/tmp/source.pdf",
                    "layout_diagnostics": {
                        "layout_profile_warning": (
                            "Proceeding with existing config/default mappings."
                        )
                    },
                }
            ),
            encoding="utf-8",
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

    assert len(page.dialogs) == 1
    snackbar = cast(ft.SnackBar, page.dialogs[0])
    message = cast(ft.Text, snackbar.content)
    assert message.value == "Proceeding with existing config/default mappings."


def test_review_selection_updates_in_place_without_remounting_editor(monkeypatch) -> None:
    monkeypatch.setattr(
        "my_ocr.ui.components.bbox_editor.get_image_size",
        lambda _path: (100, 120),
    )

    state = AppState()
    state.pages = [
        PageData(
            index=0,
            page_number=1,
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
        Callable[[object], None], cast(ft.GestureDetector, initial_stack.controls[2]).on_tap
    )

    assert len(initial_stack.controls) == 4
    select_box(SimpleNamespace())

    assert state.selected_box_id == "p0-b0"
    assert content_row.controls[1] is editor
    assert editor.content is canvas
    assert canvas.controls[0] is stack_row
    assert len(cast(ft.Stack, stack_row.controls[0]).controls) == 13
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
    assert len(cast(ft.Stack, stack_row.controls[0]).controls) == 4
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
    buttons = _toolbar_buttons_by_content(toolbar_row)
    copy_button = cast(ft.OutlinedButton, buttons["Copy OCR JSON"])
    page_markdown_button = cast(ft.Button, buttons["Download Page Markdown"])
    markdown_button = cast(ft.Button, buttons["Download OCR Markdown"])
    layout_button = cast(ft.Button, buttons["Re-detect This Page Layout"])
    rerun_button = cast(ft.Button, buttons["Re-run OCR For This Page"])
    download_button = cast(ft.Button, buttons["Download OCR JSON"])

    assert cast(ft.Text, toolbar_row.controls[4]).value == "source.pdf — OCR Complete"
    assert cast(Any, copy_button)._values["content"] == "Copy OCR JSON"
    assert copy_button.disabled is False
    assert cast(Any, page_markdown_button)._values["content"] == "Download Page Markdown"
    assert page_markdown_button.disabled is False
    assert cast(Any, markdown_button)._values["content"] == "Download OCR Markdown"
    assert markdown_button.disabled is False
    assert cast(Any, layout_button)._values["content"] == "Re-detect This Page Layout"
    assert cast(Any, rerun_button)._values["content"] == "Re-run OCR For This Page"
    assert cast(Any, download_button)._values["content"] == "Download OCR JSON"
    assert download_button.disabled is False


def test_results_view_downloads_markdown_with_md_extension(tmp_path) -> None:
    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_path = pages_dir / "page-0001.png"
    image_module.new("RGB", (100, 120), color="white").save(page_path)
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

    output_path = tmp_path / "downloaded.md"
    save_call: dict[str, Any] = {}

    class FakeFilePicker:
        async def save_file(self, **kwargs: Any) -> str:
            save_call.update(kwargs)
            return str(output_path)

    view = build_results_view(cast(ft.Page, DummyPage()), state, cast(ft.FilePicker, FakeFilePicker()))
    outer = cast(ft.Column, view.controls[0])
    inner = cast(ft.Column, outer.controls[1])
    toolbar = cast(ft.Container, inner.controls[0])
    toolbar_row = cast(ft.Row, toolbar.content)
    buttons = _toolbar_buttons_by_content(toolbar_row)
    markdown_button = cast(ft.Button, buttons["Download OCR Markdown"])

    asyncio.run(cast(Callable[[], Any], markdown_button.on_click)())

    assert save_call["file_name"] == "demo-ui.md"
    assert save_call["allowed_extensions"] == ["md"]
    assert output_path.read_text(encoding="utf-8") == "# Page 1"


def test_results_view_downloads_current_page_markdown(tmp_path) -> None:
    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_one = pages_dir / "page-0001.png"
    page_two = pages_dir / "page-0002.png"
    image_module.new("RGB", (100, 120), color="white").save(page_one)
    image_module.new("RGB", (100, 120), color="white").save(page_two)
    write_basic_ocr_outputs(
        run_dir,
        markdown="# Page 1\n\n# Page 2",
        json_payload={
            "pages": [
                build_ocr_page(
                    page_number=1,
                    page_path=str(page_one),
                    markdown="# Page 1",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"),
                ),
                build_ocr_page(
                    page_number=2,
                    page_path=str(page_two),
                    markdown="# Page 2",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json"),
                ),
            ],
            "summary": {"page_count": 2},
        },
    )

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")
    state.current_page_index = 1

    output_path = tmp_path / "page-download.md"
    save_call: dict[str, Any] = {}

    class FakeFilePicker:
        async def save_file(self, **kwargs: Any) -> str:
            save_call.update(kwargs)
            return str(output_path)

    view = build_results_view(cast(ft.Page, DummyPage()), state, cast(ft.FilePicker, FakeFilePicker()))
    outer = cast(ft.Column, view.controls[0])
    inner = cast(ft.Column, outer.controls[1])
    toolbar = cast(ft.Container, inner.controls[0])
    toolbar_row = cast(ft.Row, toolbar.content)
    buttons = _toolbar_buttons_by_content(toolbar_row)
    markdown_button = cast(ft.Button, buttons["Download Page Markdown"])

    asyncio.run(cast(Callable[[], Any], markdown_button.on_click)())

    assert save_call["file_name"] == "demo-ui-page-0002.md"
    assert save_call["allowed_extensions"] == ["md"]
    assert output_path.read_text(encoding="utf-8") == "# Page 2"


def test_results_view_disables_page_markdown_download_without_per_page_ocr_payload(tmp_path) -> None:
    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_one = pages_dir / "page-0001.png"
    page_two = pages_dir / "page-0002.png"
    image_module.new("RGB", (100, 120), color="white").save(page_one)
    image_module.new("RGB", (100, 120), color="white").save(page_two)
    write_basic_ocr_outputs(
        run_dir,
        markdown="# Whole document markdown",
        json_payload={
            "pages": [
                build_ocr_page(
                    page_number=1,
                    page_path=str(page_one),
                    markdown="# Page 1",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"),
                )
            ],
            "summary": {"page_count": 2},
        },
    )

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")
    state.current_page_index = 1

    view = build_results_view(cast(ft.Page, DummyPage()), state, ft.FilePicker())
    outer = cast(ft.Column, view.controls[0])
    inner = cast(ft.Column, outer.controls[1])
    toolbar = cast(ft.Container, inner.controls[0])
    toolbar_row = cast(ft.Row, toolbar.content)
    buttons = _toolbar_buttons_by_content(toolbar_row)
    page_markdown_button = cast(ft.Button, buttons["Download Page Markdown"])
    markdown_button = cast(ft.Button, buttons["Download OCR Markdown"])

    assert page_markdown_button.disabled is True
    assert markdown_button.disabled is False


def test_results_view_page_layout_redetect_routes_back_to_review_same_page(tmp_path, monkeypatch) -> None:
    from my_ocr import workflows

    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_one = pages_dir / "page-0001.png"
    page_two = pages_dir / "page-0002.png"
    image_module.new("RGB", (100, 120), color="white").save(page_one)
    image_module.new("RGB", (100, 120), color="white").save(page_two)
    write_basic_ocr_outputs(
        run_dir,
        markdown="# Page 1\n\n# Page 2",
        json_payload={
            "pages": [
                build_ocr_page(
                    page_number=1,
                    page_path=str(page_one),
                    markdown="# Page 1",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"),
                ),
                build_ocr_page(
                    page_number=2,
                    page_path=str(page_two),
                    markdown="# Page 2",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json"),
                ),
            ],
            "summary": {"page_count": 2},
        },
    )

    captured: dict[str, Any] = {}

    def fake_prepare_review_page_workflow(run: str, page_number: int, *, run_root: str, **kwargs) -> Path:
        captured["run"] = run
        captured["page_number"] = page_number
        captured["run_root"] = run_root
        captured["layout_profile"] = kwargs.get("layout_profile")
        return run_dir

    async def run_in_place(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(workflows, "prepare_review_page_workflow", fake_prepare_review_page_workflow)
    monkeypatch.setattr(asyncio, "to_thread", run_in_place)

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")
    state.current_page_index = 1

    page = DummyPage()
    view = build_results_view(cast(ft.Page, page), state, ft.FilePicker())
    outer = cast(ft.Column, view.controls[0])
    inner = cast(ft.Column, outer.controls[1])
    toolbar = cast(ft.Container, inner.controls[0])
    toolbar_row = cast(ft.Row, toolbar.content)
    buttons = _toolbar_buttons_by_content(toolbar_row)
    layout_button = cast(ft.Button, buttons["Re-detect This Page Layout"])

    cast(Callable[..., None], layout_button.on_click)()

    assert captured == {
        "run": "demo-ui",
        "page_number": 2,
        "run_root": str(run_root),
        "layout_profile": "auto",
    }
    assert page.route == "/review/demo-ui"
    assert state.current_page_index == 1


def test_results_view_page_ocr_rerun_reloads_same_page(tmp_path, monkeypatch) -> None:
    from my_ocr import workflows

    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_one = pages_dir / "page-0001.png"
    page_two = pages_dir / "page-0002.png"
    image_module.new("RGB", (100, 120), color="white").save(page_one)
    image_module.new("RGB", (100, 120), color="white").save(page_two)
    write_basic_ocr_outputs(
        run_dir,
        markdown="# Page 1\n\n# Page 2",
        json_payload={
            "pages": [
                build_ocr_page(
                    page_number=1,
                    page_path=str(page_one),
                    markdown="# Page 1",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"),
                ),
                build_ocr_page(
                    page_number=2,
                    page_path=str(page_two),
                    markdown="# Page 2",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json"),
                ),
            ],
            "summary": {"page_count": 2},
        },
    )

    captured: dict[str, Any] = {}

    def fake_run_reviewed_ocr_page_workflow(run: str, page_number: int, *, run_root: str, **kwargs) -> Path:
        captured["run"] = run
        captured["page_number"] = page_number
        captured["run_root"] = run_root
        captured["layout_profile"] = kwargs.get("layout_profile")
        write_basic_ocr_outputs(
            run_dir,
            markdown="# Page 1\n\n# Updated Page 2",
            json_payload={
                "pages": [
                    build_ocr_page(
                        page_number=1,
                        page_path=str(page_one),
                        markdown="# Page 1",
                        sdk_json_path=str(run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"),
                    ),
                    build_ocr_page(
                        page_number=2,
                        page_path=str(page_two),
                        markdown="# Updated Page 2",
                        sdk_json_path=str(run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json"),
                    ),
                ],
                "summary": {"page_count": 2},
            },
        )
        return run_dir

    async def run_in_place(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(workflows, "run_reviewed_ocr_page_workflow", fake_run_reviewed_ocr_page_workflow)
    monkeypatch.setattr(asyncio, "to_thread", run_in_place)

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")
    state.current_page_index = 1

    page = DummyPage()
    view = build_results_view(cast(ft.Page, page), state, ft.FilePicker())
    outer = cast(ft.Column, view.controls[0])
    inner = cast(ft.Column, outer.controls[1])
    toolbar = cast(ft.Container, inner.controls[0])
    toolbar_row = cast(ft.Row, toolbar.content)
    buttons = _toolbar_buttons_by_content(toolbar_row)
    rerun_button = cast(ft.Button, buttons["Re-run OCR For This Page"])

    cast(Callable[..., None], rerun_button.on_click)()

    assert captured == {
        "run": "demo-ui",
        "page_number": 2,
        "run_root": str(run_root),
        "layout_profile": "auto",
    }
    assert state.current_page_index == 1
    assert _markdown_pages_for_state(state) == ["# Page 1", "# Updated Page 2"]


def test_results_view_blocks_page_reruns_while_previous_rerun_is_in_flight(tmp_path, monkeypatch) -> None:
    from my_ocr import workflows

    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_one = pages_dir / "page-0001.png"
    page_two = pages_dir / "page-0002.png"
    image_module.new("RGB", (100, 120), color="white").save(page_one)
    image_module.new("RGB", (100, 120), color="white").save(page_two)
    write_basic_ocr_outputs(
        run_dir,
        markdown="# Page 1\n\n# Page 2",
        json_payload={
            "pages": [
                build_ocr_page(
                    page_number=1,
                    page_path=str(page_one),
                    markdown="# Page 1",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"),
                ),
                build_ocr_page(
                    page_number=2,
                    page_path=str(page_two),
                    markdown="# Page 2",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json"),
                ),
            ],
            "summary": {"page_count": 2},
        },
    )

    captured: dict[str, Any] = {}

    def fake_prepare_review_page_workflow(run: str, page_number: int, *, run_root: str, **kwargs) -> Path:
        captured["run"] = run
        captured["page_number"] = page_number
        captured["run_root"] = run_root
        captured["layout_profile"] = kwargs.get("layout_profile")
        return run_dir

    async def run_in_place(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(workflows, "prepare_review_page_workflow", fake_prepare_review_page_workflow)
    monkeypatch.setattr(asyncio, "to_thread", run_in_place)

    class DeferredTaskPage(DummyPage):
        def __init__(self) -> None:
            super().__init__()
            self.tasks: list[Callable[[], Any]] = []

        def run_task(self, task) -> None:  # noqa: ANN001
            self.tasks.append(task)

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")
    state.current_page_index = 1

    page = DeferredTaskPage()
    view = build_results_view(cast(ft.Page, page), state, ft.FilePicker())
    outer = cast(ft.Column, view.controls[0])
    inner = cast(ft.Column, outer.controls[1])
    toolbar = cast(ft.Container, inner.controls[0])
    toolbar_row = cast(ft.Row, toolbar.content)
    buttons = _toolbar_buttons_by_content(toolbar_row)
    layout_button = cast(ft.Button, buttons["Re-detect This Page Layout"])
    rerun_button = cast(ft.Button, buttons["Re-run OCR For This Page"])

    cast(Callable[..., None], layout_button.on_click)()
    cast(Callable[..., None], rerun_button.on_click)()

    assert len(page.tasks) == 1
    assert layout_button.disabled is True
    assert rerun_button.disabled is True

    asyncio.run(page.tasks[0]())

    assert captured == {
        "run": "demo-ui",
        "page_number": 2,
        "run_root": str(run_root),
        "layout_profile": "auto",
    }
    assert page.route == "/review/demo-ui"
    assert layout_button.disabled is False
    assert rerun_button.disabled is False


def test_results_view_reenables_page_reruns_after_rerun_failure(tmp_path, monkeypatch) -> None:
    from my_ocr import workflows

    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_one = pages_dir / "page-0001.png"
    page_two = pages_dir / "page-0002.png"
    image_module.new("RGB", (100, 120), color="white").save(page_one)
    image_module.new("RGB", (100, 120), color="white").save(page_two)
    write_basic_ocr_outputs(
        run_dir,
        markdown="# Page 1\n\n# Page 2",
        json_payload={
            "pages": [
                build_ocr_page(
                    page_number=1,
                    page_path=str(page_one),
                    markdown="# Page 1",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"),
                ),
                build_ocr_page(
                    page_number=2,
                    page_path=str(page_two),
                    markdown="# Page 2",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json"),
                ),
            ],
            "summary": {"page_count": 2},
        },
    )

    calls: list[int] = []

    def failing_prepare_review_page_workflow(
        run: str, page_number: int, *, run_root: str, **kwargs: Any
    ) -> Path:
        _ = (run, run_root, kwargs)
        calls.append(page_number)
        raise RuntimeError("boom")

    async def run_in_place(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(
        workflows,
        "prepare_review_page_workflow",
        failing_prepare_review_page_workflow,
    )
    monkeypatch.setattr(asyncio, "to_thread", run_in_place)

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")
    state.current_page_index = 1

    page = DummyPage()
    view = build_results_view(cast(ft.Page, page), state, ft.FilePicker())
    outer = cast(ft.Column, view.controls[0])
    inner = cast(ft.Column, outer.controls[1])
    toolbar = cast(ft.Container, inner.controls[0])
    toolbar_row = cast(ft.Row, toolbar.content)
    buttons = _toolbar_buttons_by_content(toolbar_row)
    layout_button = cast(ft.Button, buttons["Re-detect This Page Layout"])

    cast(Callable[..., None], layout_button.on_click)()

    assert calls == [2]
    assert layout_button.disabled is False
    assert len(page.dialogs) == 1
    snackbar = cast(ft.SnackBar, page.dialogs[0])
    message = cast(ft.Text, snackbar.content)
    assert message.value == "Page layout re-detect failed: boom"

    cast(Callable[..., None], layout_button.on_click)()

    assert calls == [2, 2]
    assert layout_button.disabled is False


def test_results_view_sparse_page_number_actions_use_actual_page_number(tmp_path, monkeypatch) -> None:
    from my_ocr import workflows

    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_five = pages_dir / "page-0005.png"
    image_module.new("RGB", (100, 120), color="white").save(page_five)
    write_basic_ocr_outputs(
        run_dir,
        markdown="# Page 5",
        json_payload={
            "pages": [
                build_ocr_page(
                    page_number=5,
                    page_path=str(page_five),
                    markdown="# Page 5",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0005" / "page-0005_model.json"),
                )
            ],
            "summary": {"page_count": 1},
        },
    )

    page_calls: dict[str, int] = {}
    save_call: dict[str, Any] = {}

    class FakeFilePicker:
        async def save_file(self, **kwargs: Any) -> str:
            save_call.update(kwargs)
            return str(tmp_path / "page-0005.md")

    def fake_prepare_review_page_workflow(run: str, page_number: int, *, run_root: str, **kwargs) -> Path:
        _ = (run, run_root, kwargs)
        page_calls["layout"] = page_number
        return run_dir

    def fake_run_reviewed_ocr_page_workflow(run: str, page_number: int, *, run_root: str, **kwargs) -> Path:
        _ = (run, run_root, kwargs)
        page_calls["ocr"] = page_number
        return run_dir

    async def run_in_place(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(workflows, "prepare_review_page_workflow", fake_prepare_review_page_workflow)
    monkeypatch.setattr(workflows, "run_reviewed_ocr_page_workflow", fake_run_reviewed_ocr_page_workflow)
    monkeypatch.setattr(asyncio, "to_thread", run_in_place)

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")

    page = DummyPage()
    view = build_results_view(cast(ft.Page, page), state, cast(ft.FilePicker, FakeFilePicker()))
    outer = cast(ft.Column, view.controls[0])
    inner = cast(ft.Column, outer.controls[1])
    toolbar = cast(ft.Container, inner.controls[0])
    toolbar_row = cast(ft.Row, toolbar.content)
    buttons = _toolbar_buttons_by_content(toolbar_row)

    assert cast(ft.Text, toolbar_row.controls[4]).value == "demo-ui — OCR Complete"
    assert cast(ft.Text, cast(ft.Row, cast(ft.Container, toolbar_row.controls[5]).content).controls[1]).value == "Page 5 / 1"

    asyncio.run(cast(Callable[[], Any], cast(ft.Button, buttons["Download Page Markdown"]).on_click)())
    cast(Callable[..., None], cast(ft.Button, buttons["Re-detect This Page Layout"]).on_click)()
    cast(Callable[..., None], cast(ft.Button, buttons["Re-run OCR For This Page"]).on_click)()

    assert save_call["file_name"] == "demo-ui-page-0005.md"
    assert page_calls == {"layout": 5, "ocr": 5}


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
