from __future__ import annotations

from dataclasses import dataclass

from my_ocr.application.dto import OcrOptions, RunId, WorkflowResult
from my_ocr.application.errors import MissingPage, OcrFailed
from my_ocr.application.ports import OcrEngine, RunStore


@dataclass(frozen=True, slots=True)
class RerunPageOcrCommand:
    run_id: RunId
    page_number: int
    options: OcrOptions = OcrOptions()


class RerunPageOcr:
    def __init__(self, run_store: RunStore, ocr_engine: OcrEngine) -> None:
        self._run_store = run_store
        self._ocr_engine = ocr_engine

    def __call__(self, command: RerunPageOcrCommand) -> WorkflowResult:
        snapshot = self._run_store.open_run(command.run_id)
        page = snapshot.page(command.page_number)
        if page is None:
            raise MissingPage(f"Page {command.page_number} not found")
        try:
            recognition = self._ocr_engine.recognize(
                [page], snapshot.review_layout, command.options
            )
        except Exception as exc:
            raise OcrFailed(f"OCR failed: {exc}") from exc
        if not recognition.result.pages:
            raise MissingPage(f"OCR returned no page {command.page_number}")
        tx = self._run_store.begin_update(command.run_id)
        try:
            tx.replace_page_ocr(
                command.page_number, recognition.result.pages[0], recognition.artifacts
            )
            return WorkflowResult(snapshot=tx.commit())
        except Exception:
            tx.rollback()
            raise

