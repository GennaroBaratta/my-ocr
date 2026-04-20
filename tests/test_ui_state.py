from __future__ import annotations

import json
from importlib import import_module

from free_doc_extract.ui.state import AppState
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
