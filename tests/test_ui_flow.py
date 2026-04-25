from __future__ import annotations

from pathlib import Path

import flet as ft

from my_ocr.runs.store import FilesystemRunStore
from my_ocr.domain import ProviderArtifacts
from my_ocr.domain import OcrPageResult, OcrRunResult, PageRef, ReviewLayout, RunId
from my_ocr.ui.components.code_display import build_code_display
from my_ocr.ui.ocr_result_text import ocr_json_text_for_state
from my_ocr.ui.state import AppState


def test_code_display_uses_loaded_v3_ocr_payload(tmp_path: Path) -> None:
    _seed_ocr_run(tmp_path / "runs", "demo")
    state = AppState()
    state.run_root = str(tmp_path / "runs")
    state.load_run("demo")

    text = ocr_json_text_for_state(state)
    display = build_code_display(state)

    assert '"pages"' in text
    assert isinstance(display, ft.Column)


def _seed_ocr_run(run_root: Path, run_id: str) -> None:
    from PIL import Image

    store = FilesystemRunStore(run_root)
    workspace = store.start_run("input.pdf", RunId(run_id))
    page_path = workspace.work_dir / "pages" / "page-0001.png"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), "white").save(page_path)
    store.publish_prepared_run(
        workspace,
        [
            PageRef(
                page_number=1,
                image_path="pages/page-0001.png",
                width=10,
                height=10,
                resolved_path=page_path,
            )
        ],
        ReviewLayout(pages=[], status="prepared"),
        ProviderArtifacts.empty(),
    )
    store.write_ocr_result_and_invalidate_extraction(
        RunId(run_id),
        OcrRunResult(
            pages=[
                OcrPageResult(
                    page_number=1,
                    image_path="pages/page-0001.png",
                    markdown="# OCR",
                    markdown_source="sdk_markdown",
                )
            ],
            markdown="# OCR",
        ),
        ProviderArtifacts.empty(),
    )



