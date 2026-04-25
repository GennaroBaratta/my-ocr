from __future__ import annotations

from dataclasses import dataclass

from my_ocr.application.dto import LayoutOptions, RunId, WorkflowResult
from my_ocr.application.errors import LayoutDetectionFailed
from my_ocr.application.ports import DocumentNormalizer, LayoutDetector, RunStore


@dataclass(frozen=True, slots=True)
class PrepareLayoutReviewCommand:
    input_path: str
    run_id: RunId | None = None
    options: LayoutOptions = LayoutOptions()


class PrepareLayoutReview:
    def __init__(
        self,
        run_store: RunStore,
        normalizer: DocumentNormalizer,
        layout_detector: LayoutDetector,
    ) -> None:
        self._run_store = run_store
        self._normalizer = normalizer
        self._layout_detector = layout_detector

    def __call__(self, command: PrepareLayoutReviewCommand) -> WorkflowResult:
        tx = self._run_store.create_run(command.input_path, command.run_id)
        try:
            pages = self._normalizer.normalize(command.input_path, tx.work_dir / "pages")
            tx.write_pages(pages)
            try:
                result = self._layout_detector.detect_layout(pages, command.options)
            except Exception as exc:
                raise LayoutDetectionFailed(f"Layout detection failed: {exc}") from exc
            tx.write_review_layout(result.layout, result.artifacts, result.diagnostics)
            tx.clear_extraction_outputs()
            snapshot = tx.commit()
            return WorkflowResult(snapshot=snapshot, warning=result.diagnostics.warning)
        except Exception:
            tx.rollback()
            raise
