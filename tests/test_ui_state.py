from __future__ import annotations

import json
from importlib import import_module

from my_ocr.ui.state import AppState, BoundingBox, PageData
from tests.support import (
    build_reviewed_layout_block,
    build_reviewed_layout_page,
)


def test_app_state_loads_and_saves_reviewed_layout(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_path = pages_dir / "page-0001.png"

    image_module = import_module("PIL.Image")
    image_module.new("RGB", (100, 120), color="white").save(page_path)

    reviewed_layout_path = run_dir / "reviewed_layout.json"
    reviewed_layout_path.write_text(
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
                        blocks=[
                            build_reviewed_layout_block(
                                label="table",
                                content="Original",
                                bbox=[10, 20, 40, 60],
                            )
                        ],
                    )
                ],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")

    assert len(state.pages) == 1
    assert len(state.pages[0].boxes) == 1
    assert state.pages[0].boxes[0].label == "table"
    assert state.pages[0].boxes[0].x == 10
    assert state.pages[0].boxes[0].height == 40

    state.pages[0].boxes[0].x = 12
    state.pages[0].boxes[0].width = 35
    state.save_reviewed_layout()

    persisted = json.loads(reviewed_layout_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "reviewed"
    assert persisted["pages"][0]["blocks"][0]["bbox"] == [12, 20, 47, 60]
    assert persisted["pages"][0]["blocks"][0]["label"] == "table"


def test_load_run_resets_add_box_mode(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_path = pages_dir / "page-0001.png"

    image_module = import_module("PIL.Image")
    image_module.new("RGB", (100, 120), color="white").save(page_path)

    reviewed_layout_path = run_dir / "reviewed_layout.json"
    reviewed_layout_path.write_text(
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

    state = AppState()
    state.run_root = str(run_root)
    state.is_adding_box = True

    state.load_run("demo-ui")

    assert state.is_adding_box is False


def test_save_reviewed_layout_preserves_sparse_page_number(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)

    page_path = pages_dir / "page-0005.png"

    image_module = import_module("PIL.Image")
    image_module.new("RGB", (100, 120), color="white").save(page_path)

    reviewed_layout_path = run_dir / "reviewed_layout.json"
    reviewed_layout_path.write_text(
        json.dumps(
            {
                "version": 1,
                "status": "reviewed",
                "pages": [
                    build_reviewed_layout_page(
                        page_number=5,
                        page_path=str(page_path),
                        source_sdk_json_path=str(
                            run_dir / "ocr_raw" / "page-0005" / "page-0005_model.json"
                        ),
                        blocks=[build_reviewed_layout_block(content="Sparse page")],
                    )
                ],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )

    state = AppState()
    state.run_root = str(run_root)
    state.load_run("demo-ui")
    state.save_reviewed_layout()

    persisted = json.loads(reviewed_layout_path.read_text(encoding="utf-8"))
    assert state.current_page_number == 5
    assert persisted["pages"][0]["page_number"] == 5


def test_run_root_setter_updates_repository_and_recent_run_reads(tmp_path) -> None:
    first_root = tmp_path / "first-runs"
    second_root = tmp_path / "second-runs"
    run_dir = second_root / "demo-ui"
    (run_dir / "pages").mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"input_path": "data/raw/demo.pdf"}),
        encoding="utf-8",
    )

    state = AppState()
    state.run_root = str(first_root)
    state.load_recent_runs()
    assert state.recent_runs == []

    state.run_root = str(second_root)
    state.load_recent_runs()

    assert state.repository.run_root == str(second_root)
    assert [run.run_id for run in state.recent_runs] == ["demo-ui"]


def test_load_run_resets_page_index_and_selected_box(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_path = pages_dir / "page-0001.png"

    image_module = import_module("PIL.Image")
    image_module.new("RGB", (100, 120), color="white").save(page_path)

    reviewed_layout_path = run_dir / "reviewed_layout.json"
    reviewed_layout_path.write_text(
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
                        blocks=[build_reviewed_layout_block()],
                    )
                ],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )

    state = AppState()
    state.run_root = str(run_root)
    state.current_page_index = 7
    state.selected_box_id = "p0-b0"

    state.load_run("demo-ui")

    assert state.current_page_index == 0
    assert state.selected_box_id is None


def test_box_mutations_delegate_review_saves(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_path = pages_dir / "page-0001.png"

    image_module = import_module("PIL.Image")
    image_module.new("RGB", (100, 120), color="white").save(page_path)

    state = AppState()
    state.run_root = str(run_root)
    state.pages = [
        PageData(
            index=0,
            page_number=1,
            image_path=str(page_path),
            boxes=[
                build_box("p0-b0"),
            ],
        )
    ]

    save_calls: list[tuple[object, object]] = []

    def fake_save_reviewed_layout(run_paths, pages) -> None:
        save_calls.append((run_paths, pages))

    monkeypatch.setattr(state.repository, "save_reviewed_layout", fake_save_reviewed_layout)

    state.update_box("p0-b0", label="table")
    new_box_id = state.add_box_to_current_page()
    assert new_box_id is not None
    state.remove_box(new_box_id)

    assert len(save_calls) == 3


def build_box(box_id: str):
    return BoundingBox(
        id=box_id,
        page_index=0,
        x=1,
        y=2,
        width=10,
        height=20,
        label="text",
    )
