from __future__ import annotations

from pathlib import Path

import flet as ft

from my_ocr.adapters.outbound.filesystem.run_store import FilesystemRunStore
from my_ocr.application.artifacts import ProviderArtifacts
from my_ocr.models import OcrPageResult, OcrRunResult, PageRef, RunId
from my_ocr.ui.components.code_display import build_code_display
from my_ocr.ui.ocr_result_text import ocr_json_text_for_state
from my_ocr.ui.state import AppState


def test_code_display_uses_loaded_v2_ocr_payload(tmp_path: Path) -> None:
    _seed_ocr_run(tmp_path / "runs", "demo")
    state = AppState()
    state.run_root = str(tmp_path / "runs")
    state.load_run("demo")

    text = ocr_json_text_for_state(state)
    display = build_code_display(state)

    assert '"page_count": 1' in text
    assert isinstance(display, ft.Column)


def test_unsupported_run_message_is_loaded_for_v1_folder(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "old"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text("{}", encoding="utf-8")
    state = AppState()
    state.run_root = str(tmp_path / "runs")

    state.load_run("old")

    assert state.session.unsupported_run_message
    assert "created before v2" in state.session.unsupported_run_message


def _seed_ocr_run(run_root: Path, run_id: str) -> None:
    from PIL import Image

    store = FilesystemRunStore(run_root)
    tx = store.create_run("input.pdf", RunId(run_id))
    page_path = tx.work_dir / "pages" / "page-0001.png"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), "white").save(page_path)
    tx.write_pages([PageRef(1, "pages/page-0001.png", 10, 10, page_path)])
    tx.write_ocr_result(
        OcrRunResult(
            [OcrPageResult(1, "pages/page-0001.png", "# OCR", "sdk_markdown")],
            "# OCR",
        ),
        ProviderArtifacts.empty(),
    )
    tx.commit()


