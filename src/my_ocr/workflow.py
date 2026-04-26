from __future__ import annotations

from my_ocr.domain import (
    OcrRuntimeOptions,
    ReviewLayout,
    RunId,
    StructuredExtractionOptions,
    WorkflowResult,
)
from .use_cases import (
    DocumentNormalizer,
    ExtractionUseCase,
    LayoutDetector,
    OcrEngine,
    OcrUseCase,
    ReviewUseCase,
    RulesExtractor,
    RunRepository,
    StructuredExtractor,
)


class DocumentWorkflow:
    """Public facade for document processing workflows."""

    def __init__(
        self,
        run_store: RunRepository,
        normalizer: DocumentNormalizer,
        layout_detector: LayoutDetector,
        ocr_engine: OcrEngine,
        rules_extractor: RulesExtractor,
        structured_extractor: StructuredExtractor,
    ) -> None:
        self._review = ReviewUseCase(run_store, normalizer, layout_detector)
        self._ocr = OcrUseCase(run_store, ocr_engine)
        self._extraction = ExtractionUseCase(run_store, rules_extractor, structured_extractor)

    def prepare_review(
        self,
        input_path: str,
        run_id: RunId | None = None,
        options: OcrRuntimeOptions = OcrRuntimeOptions(),
    ) -> WorkflowResult:
        return self._review.prepare_review(input_path, run_id, options)

    def run_reviewed_ocr(
        self,
        run_id: RunId,
        options: OcrRuntimeOptions = OcrRuntimeOptions(),
    ) -> WorkflowResult:
        return self._ocr.run_reviewed_ocr(run_id, options)

    def save_review_layout(self, run_id: RunId, layout: ReviewLayout) -> WorkflowResult:
        return self._review.save_review_layout(run_id, layout)

    def rerun_page_layout(
        self,
        run_id: RunId,
        page_number: int,
        options: OcrRuntimeOptions = OcrRuntimeOptions(),
    ) -> WorkflowResult:
        return self._review.rerun_page_layout(run_id, page_number, options)

    def rerun_page_ocr(
        self,
        run_id: RunId,
        page_number: int,
        options: OcrRuntimeOptions = OcrRuntimeOptions(),
    ) -> WorkflowResult:
        return self._ocr.rerun_page_ocr(run_id, page_number, options)

    def extract_rules(self, run_id: RunId) -> WorkflowResult:
        return self._extraction.extract_rules(run_id)

    def extract_structured(
        self,
        run_id: RunId,
        options: StructuredExtractionOptions = StructuredExtractionOptions(),
    ) -> WorkflowResult:
        return self._extraction.extract_structured(run_id, options)

    def run_automatic(
        self,
        input_path: str,
        run_id: RunId | None = None,
        layout_options: OcrRuntimeOptions = OcrRuntimeOptions(),
        ocr_options: OcrRuntimeOptions = OcrRuntimeOptions(),
    ) -> WorkflowResult:
        prepared = self.prepare_review(
            input_path=input_path,
            run_id=run_id,
            options=layout_options,
        )
        resolved_run_id = prepared.snapshot.run_id
        self.run_reviewed_ocr(resolved_run_id, options=ocr_options)
        return self.extract_rules(resolved_run_id)


__all__ = [
    "DocumentWorkflow",
    "DocumentNormalizer",
    "LayoutDetector",
    "OcrEngine",
    "RulesExtractor",
    "RunRepository",
    "StructuredExtractor",
]
