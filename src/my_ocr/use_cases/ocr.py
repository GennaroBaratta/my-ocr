from __future__ import annotations

from my_ocr.domain import (
    MissingPage,
    OcrFailed,
    OcrPageResult,
    OcrRunResult,
    OcrRuntimeOptions,
    RunId,
    RunSnapshot,
    WorkflowResult,
)
from .ports import OcrEngine, RunRepository


class OcrUseCase:
    def __init__(self, run_store: RunRepository, ocr_engine: OcrEngine) -> None:
        self._run_store = run_store
        self._ocr_engine = ocr_engine

    def run_reviewed_ocr(
        self,
        run_id: RunId,
        options: OcrRuntimeOptions = OcrRuntimeOptions(),
    ) -> WorkflowResult:
        snapshot = self._run_store.open_run(run_id)
        if not snapshot.pages:
            raise MissingPage(f"No pages found for run {run_id}")
        try:
            recognition = self._ocr_engine.recognize(
                snapshot.pages, snapshot.run_dir, snapshot.review_layout, options
            )
        except Exception as exc:
            raise OcrFailed(f"OCR failed: {exc}") from exc
        snapshot = self._run_store.write_ocr_result_and_invalidate_extraction(
            run_id, recognition.result, recognition.artifacts
        )
        return WorkflowResult(snapshot=snapshot)

    def rerun_page_ocr(
        self,
        run_id: RunId,
        page_number: int,
        options: OcrRuntimeOptions = OcrRuntimeOptions(),
    ) -> WorkflowResult:
        snapshot = self._run_store.open_run(run_id)
        page = snapshot.page(page_number)
        if page is None:
            raise MissingPage(f"Page {page_number} not found")
        try:
            recognition = self._ocr_engine.recognize(
                [page], snapshot.run_dir, snapshot.review_layout, options
            )
        except Exception as exc:
            raise OcrFailed(f"OCR failed: {exc}") from exc
        if not recognition.result.pages:
            raise MissingPage(f"OCR returned no page {page_number}")
        result = _replace_ocr_page(snapshot, page_number, recognition.result.pages[0])
        snapshot = self._run_store.write_ocr_result_and_invalidate_extraction(
            run_id, result, recognition.artifacts
        )
        return WorkflowResult(snapshot=snapshot)


def _replace_ocr_page(
    snapshot: RunSnapshot, page_number: int, page: OcrPageResult
) -> OcrRunResult:
    if snapshot.ocr_result is None:
        raise MissingPage("Cannot replace a page before OCR has been run.")
    pages_by_number = {ocr_page.page_number: ocr_page for ocr_page in snapshot.ocr_result.pages}
    pages_by_number[page_number] = page
    ordered_pages = [
        pages_by_number[manifest_page.page_number]
        for manifest_page in snapshot.pages
        if manifest_page.page_number in pages_by_number
    ]
    markdown = "\n\n".join(page.markdown.strip() for page in ordered_pages if page.markdown.strip())
    return OcrRunResult(
        pages=ordered_pages,
        markdown=markdown,
        diagnostics=snapshot.ocr_result.diagnostics,
    )


__all__ = ["OcrUseCase"]
