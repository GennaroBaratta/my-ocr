from __future__ import annotations

from pathlib import Path
from typing import cast

import flet as ft

from my_ocr.runs.store import FilesystemRunStore
from my_ocr.domain import ProviderArtifacts
from my_ocr.domain import OcrPageResult, OcrRunResult, PageRef, ReviewLayout, RunId
from my_ocr.ui.components.code_display import build_code_display
from my_ocr.ui.ocr_result_text import current_page_ocr_markdown_for_state, ocr_json_text_for_state
from my_ocr.ui.features.results.actions import ResultsScreenActions
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


def test_results_flow_uses_ocr_payload_when_structured_extraction_is_absent(
    tmp_path: Path,
) -> None:
    _seed_ocr_run(tmp_path / "runs", "demo")
    state = AppState()
    state.run_root = str(tmp_path / "runs")

    state.load_run("demo")
    text = ocr_json_text_for_state(state)
    display = build_code_display(state)

    assert not (tmp_path / "runs" / "demo" / "extraction").exists()
    assert current_page_ocr_markdown_for_state(state) == "# OCR"
    assert '"markdown": "# OCR"' in text
    assert "document_type" not in text
    assert state.session.extraction_json == {}
    assert isinstance(display, ft.Column)


def test_results_page_rerun_reload_preserves_active_page(tmp_path: Path) -> None:
    _seed_ocr_run(tmp_path / "runs", "demo", page_count=3)
    state = AppState()
    state.run_root = str(tmp_path / "runs")
    state.load_run("demo")
    state.set_current_page_index(1)
    actions = ResultsScreenActions(
        page=cast(ft.Page, object()),
        state=state,
        file_picker=cast(ft.FilePicker, object()),
        rebuild=lambda: None,
        current_ocr_json_text=lambda: "{}",
        current_ocr_markdown_text=lambda: "# OCR",
        current_page_export_markdown_text=lambda: "# OCR",
    )

    actions._reload_state(state.session.current_page_index)

    assert state.session.current_page_index == 1
    assert state.current_page_number == 2


def _seed_ocr_run(run_root: Path, run_id: str, *, page_count: int = 1) -> None:
    store = FilesystemRunStore(run_root)
    workspace = store.start_run("input.pdf", RunId(run_id))
    pages: list[PageRef] = []
    ocr_pages: list[OcrPageResult] = []
    for page_number in range(1, page_count + 1):
        page_path = workspace.work_dir / "pages" / f"page-{page_number:04d}.png"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_bytes(_PNG_BYTES)
        pages.append(
            PageRef(
                page_number=page_number,
                image_path=f"pages/page-{page_number:04d}.png",
                width=10,
                height=10,
                resolved_path=page_path,
            )
        )
        ocr_pages.append(
            OcrPageResult(
                page_number=page_number,
                image_path=f"pages/page-{page_number:04d}.png",
                markdown="# OCR",
                markdown_source="sdk_markdown",
            )
        )
    store.publish_prepared_run(
        workspace,
        pages,
        ReviewLayout(pages=[], status="prepared"),
        ProviderArtifacts.empty(),
    )
    store.write_ocr_result_and_invalidate_extraction(
        RunId(run_id),
        OcrRunResult(
            pages=ocr_pages,
            markdown="# OCR",
        ),
        ProviderArtifacts.empty(),
    )


_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc``\x00\x00\x00\x04\x00\x01\xf6\x178U"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)
