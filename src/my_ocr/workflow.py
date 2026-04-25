from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from my_ocr.domain import (
    LayoutDetectionResult,
    LayoutDiagnostics,
    OcrPageResult,
    OcrRecognitionResult,
    OcrRunResult,
    ProviderArtifacts,
    WorkflowResult,
    LayoutDetectionFailed,
    MissingPage,
    OcrFailed,
    StructuredExtractionFailed,
    PageRef,
    ReviewLayout,
    ReviewPage,
    RunId,
    RunSnapshot,
    OcrRuntimeOptions,
    StructuredExtractionOptions,
)
from my_ocr.extraction.canonical import choose_canonical_prediction


class RunWorkspace(Protocol):
    run_id: RunId
    target_dir: Path
    work_dir: Path


class RunRepository(Protocol):
    def start_run(
        self, input_path: str | Path, run_id: RunId | None = None
    ) -> RunWorkspace: ...

    def publish_prepared_run(
        self,
        workspace: RunWorkspace,
        pages: list[PageRef],
        layout: ReviewLayout,
        artifacts: ProviderArtifacts,
        diagnostics: LayoutDiagnostics | None = None,
    ) -> RunSnapshot: ...

    def discard_workspace(self, workspace: RunWorkspace) -> None: ...

    def open_run(self, run_id: RunId | str) -> RunSnapshot: ...

    def save_review_layout_and_invalidate_downstream(
        self,
        run_id: RunId | str,
        layout: ReviewLayout,
        artifacts: ProviderArtifacts,
        diagnostics: LayoutDiagnostics | None = None,
    ) -> RunSnapshot: ...

    def write_ocr_result_and_invalidate_extraction(
        self,
        run_id: RunId | str,
        result: OcrRunResult,
        artifacts: ProviderArtifacts,
    ) -> RunSnapshot: ...

    def write_rules_extraction(
        self, run_id: RunId | str, prediction: dict[str, Any]
    ) -> RunSnapshot: ...

    def write_structured_extraction(
        self,
        run_id: RunId | str,
        prediction: dict[str, Any],
        metadata: dict[str, Any],
        *,
        canonical_prediction: dict[str, Any],
    ) -> RunSnapshot: ...


class DocumentNormalizer(Protocol):
    def __call__(self, input_path: str | Path, pages_dir: str | Path) -> list[PageRef]: ...


class LayoutDetector(Protocol):
    def detect_layout(
        self, pages: list[PageRef], run_dir: Path, options: OcrRuntimeOptions
    ) -> LayoutDetectionResult: ...


class OcrEngine(Protocol):
    def recognize(
        self,
        pages: list[PageRef],
        run_dir: Path,
        review: ReviewLayout | None,
        options: OcrRuntimeOptions,
    ) -> OcrRecognitionResult: ...


class RulesExtractor(Protocol):
    def __call__(self, markdown: str) -> dict[str, Any]: ...


class StructuredExtractor(Protocol):
    def extract(
        self,
        pages: list[PageRef],
        *,
        markdown_text: str | None,
        options: StructuredExtractionOptions,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


class DocumentWorkflow:
    """Coordinates document processing."""

    def __init__(
        self,
        run_store: RunRepository,
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
        options: OcrRuntimeOptions = OcrRuntimeOptions(),
    ) -> WorkflowResult:
        workspace = self._run_store.start_run(input_path, run_id)
        try:
            pages = self._normalizer(input_path, workspace.work_dir / "pages")
            try:
                result = self._layout_detector.detect_layout(pages, workspace.work_dir, options)
            except Exception as exc:
                raise LayoutDetectionFailed(f"Layout detection failed: {exc}") from exc
            snapshot = self._run_store.publish_prepared_run(
                workspace,
                pages,
                result.layout,
                result.artifacts,
                result.diagnostics,
            )
            return WorkflowResult(snapshot=snapshot, warning=result.diagnostics.warning)
        except Exception:
            self._run_store.discard_workspace(workspace)
            raise

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

    def save_review_layout(self, run_id: RunId, layout: ReviewLayout) -> WorkflowResult:
        snapshot = self._run_store.save_review_layout_and_invalidate_downstream(
            run_id, layout, ProviderArtifacts.empty()
        )
        assert snapshot is not None
        return WorkflowResult(snapshot=snapshot)

    def rerun_page_layout(
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
            result = self._layout_detector.detect_layout([page], snapshot.run_dir, options)
        except Exception as exc:
            raise LayoutDetectionFailed(f"Layout detection failed: {exc}") from exc
        if not result.layout.pages:
            raise MissingPage(f"Layout detector returned no page {page_number}")
        layout = _replace_review_page(snapshot, page_number, result.layout.pages[0])
        snapshot = self._run_store.save_review_layout_and_invalidate_downstream(
            run_id,
            layout,
            result.artifacts,
            result.diagnostics,
        )
        return WorkflowResult(snapshot=snapshot, warning=result.diagnostics.warning)

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

    def extract_rules(self, run_id: RunId) -> WorkflowResult:
        snapshot = self._run_store.open_run(run_id)
        if snapshot.ocr_result is None or not snapshot.ocr_result.markdown.strip():
            raise OcrFailed("Rules extraction requires OCR markdown.")
        prediction = self._rules_extractor(snapshot.ocr_result.markdown)
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

        rules_prediction = snapshot.extraction.get("rules")
        canonical, canonical_metadata = choose_canonical_prediction(
            structured_prediction=prediction,
            rules_prediction=rules_prediction if isinstance(rules_prediction, dict) else None,
            source_text=markdown,
        )
        metadata = {**metadata, **canonical_metadata}

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


def _replace_review_page(
    snapshot: RunSnapshot, page_number: int, page: ReviewPage
) -> ReviewLayout:
    existing = snapshot.review_layout or ReviewLayout(pages=[], status="prepared")
    pages_by_number = {review_page.page_number: review_page for review_page in existing.pages}
    pages_by_number[page_number] = page
    ordered_pages = [
        pages_by_number[manifest_page.page_number]
        for manifest_page in snapshot.pages
        if manifest_page.page_number in pages_by_number
    ]
    return ReviewLayout(pages=ordered_pages, status="prepared", version=existing.version)


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

