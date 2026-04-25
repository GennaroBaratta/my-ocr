from __future__ import annotations

import json
from pathlib import Path

from my_ocr.adapters.outbound.filesystem.run_store import FilesystemRunStore
from my_ocr.application.dto import PageRef, RunId
from my_ocr.ui.state import AppState


def test_app_state_loads_v2_run_and_saves_reviewed_layout(tmp_path: Path) -> None:
    _seed_run(tmp_path / "runs", "demo")
    state = AppState()
    state.run_root = str(tmp_path / "runs")

    state.load_run("demo")
    box_id = state.add_box_to_current_page(label="table", x=1, y=2, width=3, height=4)

    persisted = json.loads(
        (tmp_path / "runs" / "demo" / "layout" / "review.json").read_text(encoding="utf-8")
    )
    assert box_id
    assert state.run_id == "demo"
    assert persisted["status"] == "reviewed"
    assert persisted["pages"][0]["blocks"][0]["label"] == "table"


def test_load_run_resets_selection_and_add_box_mode(tmp_path: Path) -> None:
    _seed_run(tmp_path / "runs", "demo")
    state = AppState()
    state.run_root = str(tmp_path / "runs")
    state.is_adding_box = True
    state.current_page_index = 4
    state.selected_box_id = "missing"

    state.load_run("demo")

    assert state.is_adding_box is False
    assert state.current_page_index == 0
    assert state.selected_box_id is None


def test_run_root_setter_rebuilds_services_for_recent_runs(tmp_path: Path) -> None:
    _seed_run(tmp_path / "runs-a", "a")
    _seed_run(tmp_path / "runs-b", "b")
    state = AppState()

    state.run_root = str(tmp_path / "runs-a")
    state.load_recent_runs()
    assert [run.run_id for run in state.recent_runs] == ["a"]

    state.run_root = str(tmp_path / "runs-b")
    state.load_recent_runs()
    assert [run.run_id for run in state.recent_runs] == ["b"]


def _seed_run(run_root: Path, run_id: str) -> None:
    from PIL import Image

    store = FilesystemRunStore(run_root)
    tx = store.create_run("input.pdf", RunId(run_id))
    page_path = tx.work_dir / "pages" / "page-0001.png"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), "white").save(page_path)
    tx.write_pages(
        [
            PageRef(
                page_number=1,
                image_path="pages/page-0001.png",
                width=10,
                height=10,
                resolved_path=page_path,
            )
        ]
    )
    tx.commit()

