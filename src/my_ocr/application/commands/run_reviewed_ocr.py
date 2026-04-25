from __future__ import annotations

from dataclasses import dataclass

from my_ocr.application.dto import OcrOptions, RunId, WorkflowResult
from my_ocr.application.errors import MissingPage, OcrFailed
from my_ocr.application.ports import OcrEngine, RunStore


@dataclass(frozen=True, slots=True)
class RunReviewedOcrCommand:
    run_id: RunId
    options: OcrOptions = OcrOptions()


class RunReviewedOcr:
    def __init__(self, run_store: RunStore, ocr_engine: OcrEngine) -> None:
        self._run_store = run_store
        self._ocr_engine = ocr_engine

    def __call__(self, command: RunReviewedOcrCommand) -> WorkflowResult:
        snapshot = self._run_store.open_run(command.run_id)
        if not snapshot.pages:
            raise MissingPage(f"No pages found for run {command.run_id}")
        try:
            recognition = self._ocr_engine.recognize(
                snapshot.pages, snapshot.review_layout, command.options
            )
        except Exception as exc:
            raise OcrFailed(f"OCR failed: {exc}") from exc
        tx = self._run_store.begin_update(command.run_id)
        try:
            tx.write_ocr_result(recognition.result, recognition.artifacts)
            return WorkflowResult(snapshot=tx.commit())
        except Exception:
            tx.rollback()
            raise

