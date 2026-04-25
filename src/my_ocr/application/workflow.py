from __future__ import annotations

from my_ocr.application.artifacts import ProviderArtifacts
from my_ocr.application.models import ReviewLayout, RunId
from my_ocr.application.options import (
    LayoutOptions,
    OcrOptions,
    StructuredExtractionOptions,
)
from my_ocr.application.results import WorkflowResult
from my_ocr.application.errors import (
    LayoutDetectionFailed,
    MissingPage,
    OcrFailed,
    StructuredExtractionFailed,
)
from my_ocr.application.ports import (
    DocumentNormalizer,
    LayoutDetector,
    OcrEngine,
    RulesExtractor,
    RunStore,
    StructuredExtractor,
)
from my_ocr.application.services.structured_validation import validate_structured_prediction


class DocumentWorkflow:
    """Coordinates the document processing workflow against application ports."""

    def __init__(
        self,
        run_store: RunStore,
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
        tx = self._run_store.create_run(input_path, run_id)
        try:
            pages = self._normalizer.normalize(input_path, tx.work_dir / "pages")
            tx.write_pages(pages)
            try:
                result = self._layout_detector.detect_layout(pages, options)
            except Exception as exc:
                raise LayoutDetectionFailed(f"Layout detection failed: {exc}") from exc
            tx.write_review_layout(result.layout, result.artifacts, result.diagnostics)
            tx.clear_extraction_outputs()
            snapshot = tx.commit()
            return WorkflowResult(snapshot=snapshot, warning=result.diagnostics.warning)
        except Exception:
            tx.rollback()
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
        tx = self._run_store.begin_update(run_id)
        try:
            tx.write_ocr_result(recognition.result, recognition.artifacts)
            return WorkflowResult(snapshot=tx.commit())
        except Exception:
            tx.rollback()
            raise

    def save_review_layout(self, run_id: RunId, layout: ReviewLayout) -> WorkflowResult:
        tx = self._run_store.begin_update(run_id)
        try:
            tx.write_review_layout(layout, ProviderArtifacts.empty())
            return WorkflowResult(snapshot=tx.commit())
        except Exception:
            tx.rollback()
            raise

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
        tx = self._run_store.begin_update(run_id)
        try:
            tx.replace_page_layout(page_number, result.layout.pages[0], result.artifacts)
            snapshot = tx.commit()
            return WorkflowResult(snapshot=snapshot, warning=result.diagnostics.warning)
        except Exception:
            tx.rollback()
            raise

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
        tx = self._run_store.begin_update(run_id)
        try:
            tx.replace_page_ocr(page_number, recognition.result.pages[0], recognition.artifacts)
            return WorkflowResult(snapshot=tx.commit())
        except Exception:
            tx.rollback()
            raise

    def extract_rules(self, run_id: RunId) -> WorkflowResult:
        snapshot = self._run_store.open_run(run_id)
        if snapshot.ocr_result is None or not snapshot.ocr_result.markdown.strip():
            raise OcrFailed("Rules extraction requires OCR markdown.")
        prediction = self._rules_extractor.extract(snapshot.ocr_result.markdown)
        tx = self._run_store.begin_update(run_id)
        try:
            tx.write_rules_extraction(prediction)
            return WorkflowResult(snapshot=tx.commit())
        except Exception:
            tx.rollback()
            raise

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

        tx = self._run_store.begin_update(run_id)
        try:
            tx.write_structured_extraction(
                prediction,
                metadata,
                canonical_prediction=canonical,
            )
            return WorkflowResult(snapshot=tx.commit())
        except Exception:
            tx.rollback()
            raise

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
