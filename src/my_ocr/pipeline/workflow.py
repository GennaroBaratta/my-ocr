from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from my_ocr.application.artifacts import (
    LayoutDetectionResult,
    OcrRecognitionResult,
    ProviderArtifacts,
)
from my_ocr.pipeline.errors import (
    LayoutDetectionFailed,
    MissingPage,
    OcrFailed,
    StructuredExtractionFailed,
)
from my_ocr.models import (
    PageRef,
    ReviewLayout,
    RunId,
)
from my_ocr.pipeline.options import (
    LayoutOptions,
    OcrOptions,
    StructuredExtractionOptions,
)
from my_ocr.application.results import WorkflowResult
from my_ocr.pipeline.extraction import validate_structured_prediction
from my_ocr.pipeline.storage import RunHandle, RunStorage


RunTransaction = RunHandle
RunStore = RunStorage


class DocumentNormalizer(Protocol):
    def normalize(self, input_path: str | Path, pages_dir: str | Path) -> list[PageRef]: ...


class LayoutDetector(Protocol):
    def detect_layout(
        self, pages: list[PageRef], options: LayoutOptions
    ) -> LayoutDetectionResult: ...


class OcrEngine(Protocol):
    def recognize(
        self,
        pages: list[PageRef],
        review: ReviewLayout | None,
        options: OcrOptions,
    ) -> OcrRecognitionResult: ...


class RulesExtractor(Protocol):
    def extract(self, markdown: str) -> dict[str, Any]: ...


class StructuredExtractor(Protocol):
    def extract(
        self,
        pages: list[PageRef],
        *,
        markdown_text: str | None,
        options: StructuredExtractionOptions,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


class DocumentWorkflow:
    """Coordinates the document processing workflow against application ports."""

    def __init__(
        self,
        run_store: RunStorage,
        normalizer: DocumentNormalizer,
        layout_detector: LayoutDetector,
        ocr_engine: OcrEngine,
        rules_extractor: RulesExtractor,
        structured_extractor: StructuredExtractor,
    ) -> None:
        self._run_store = run_store
        self._normalizer = normalizer
        self._layout_detector = layout_detector
        self._ocr_engine = ocr_engine
        self._rules_extractor = rules_extractor
        self._structured_extractor = structured_extractor

    def prepare_review(
        self,
        input_path: str,
        run_id: RunId | None = None,
        options: LayoutOptions = LayoutOptions(),
    ) -> WorkflowResult:
        run = self._run_store.create_run(input_path, run_id)
        try:
            pages = self._normalizer.normalize(input_path, run.work_dir / "pages")
            self._run_store.write_pages(run, pages)
            try:
                result = self._layout_detector.detect_layout(pages, options)
            except Exception as exc:
                raise LayoutDetectionFailed(f"Layout detection failed: {exc}") from exc
            self._run_store.write_review_layout(
                run, result.layout, result.artifacts, result.diagnostics
            )
            self._run_store.clear_extraction_outputs(run)
            snapshot = self._run_store.commit_run(run)
            return WorkflowResult(snapshot=snapshot, warning=result.diagnostics.warning)
        except Exception:
            self._run_store.rollback_run(run)
            raise

    def run_reviewed_ocr(
        self,
        run_id: RunId,
        options: OcrOptions = OcrOptions(),
    ) -> WorkflowResult:
        snapshot = self._run_store.open_run(run_id)
        if not snapshot.pages:
            raise MissingPage(f"No pages found for run {run_id}")
        try:
            recognition = self._ocr_engine.recognize(
                snapshot.pages, snapshot.review_layout, options
            )
        except Exception as exc:
            raise OcrFailed(f"OCR failed: {exc}") from exc
        snapshot = self._run_store.write_ocr_result(
            run_id, recognition.result, recognition.artifacts
        )
        return WorkflowResult(snapshot=snapshot)

    def save_review_layout(self, run_id: RunId, layout: ReviewLayout) -> WorkflowResult:
        snapshot = self._run_store.write_review_layout(
            run_id, layout, ProviderArtifacts.empty()
        )
        assert snapshot is not None
        return WorkflowResult(snapshot=snapshot)

    def rerun_page_layout(
        self,
        run_id: RunId,
        page_number: int,
        options: LayoutOptions = LayoutOptions(),
    ) -> WorkflowResult:
        snapshot = self._run_store.open_run(run_id)
        page = snapshot.page(page_number)
        if page is None:
            raise MissingPage(f"Page {page_number} not found")
        try:
            result = self._layout_detector.detect_layout([page], options)
        except Exception as exc:
            raise LayoutDetectionFailed(f"Layout detection failed: {exc}") from exc
        if not result.layout.pages:
            raise MissingPage(f"Layout detector returned no page {page_number}")
        snapshot = self._run_store.replace_page_layout(
            run_id, page_number, result.layout.pages[0], result.artifacts
        )
        return WorkflowResult(snapshot=snapshot, warning=result.diagnostics.warning)

    def rerun_page_ocr(
        self,
        run_id: RunId,
        page_number: int,
        options: OcrOptions = OcrOptions(),
    ) -> WorkflowResult:
        snapshot = self._run_store.open_run(run_id)
        page = snapshot.page(page_number)
        if page is None:
            raise MissingPage(f"Page {page_number} not found")
        try:
            recognition = self._ocr_engine.recognize([page], snapshot.review_layout, options)
        except Exception as exc:
            raise OcrFailed(f"OCR failed: {exc}") from exc
        if not recognition.result.pages:
            raise MissingPage(f"OCR returned no page {page_number}")
        snapshot = self._run_store.replace_page_ocr(
            run_id, page_number, recognition.result.pages[0], recognition.artifacts
        )
        return WorkflowResult(snapshot=snapshot)

    def extract_rules(self, run_id: RunId) -> WorkflowResult:
        snapshot = self._run_store.open_run(run_id)
        if snapshot.ocr_result is None or not snapshot.ocr_result.markdown.strip():
            raise OcrFailed("Rules extraction requires OCR markdown.")
        prediction = self._rules_extractor.extract(snapshot.ocr_result.markdown)
        snapshot = self._run_store.write_rules_extraction(run_id, prediction)
        return WorkflowResult(snapshot=snapshot)

    def extract_structured(
        self,
        run_id: RunId,
        options: StructuredExtractionOptions = StructuredExtractionOptions(),
    ) -> WorkflowResult:
        snapshot = self._run_store.open_run(run_id)
        markdown = snapshot.ocr_result.markdown if snapshot.ocr_result else None
        try:
            prediction, metadata = self._structured_extractor.extract(
                snapshot.pages,
                markdown_text=markdown,
                options=options,
            )
        except Exception as exc:
            raise StructuredExtractionFailed(f"Structured extraction failed: {exc}") from exc

        validation = validate_structured_prediction(prediction, source_text=markdown)
        canonical = prediction
        canonical_source = "structured"
        rules_prediction = snapshot.extraction.get("rules")
        if not validation["ok"] and isinstance(rules_prediction, dict):
            canonical = rules_prediction
            canonical_source = "rules"
        metadata = {**metadata, "canonical_source": canonical_source, "validation": validation}

        snapshot = self._run_store.write_structured_extraction(
            run_id,
            prediction,
            metadata,
            canonical_prediction=canonical,
        )
        return WorkflowResult(snapshot=snapshot)

    def run_automatic(
        self,
        input_path: str,
        run_id: RunId | None = None,
        layout_options: LayoutOptions = LayoutOptions(),
        ocr_options: OcrOptions = OcrOptions(),
    ) -> WorkflowResult:
        prepared = self.prepare_review(
            input_path=input_path,
            run_id=run_id,
            options=layout_options,
        )
        resolved_run_id = prepared.snapshot.run_id
        self.run_reviewed_ocr(resolved_run_id, options=ocr_options)
        return self.extract_rules(resolved_run_id)
