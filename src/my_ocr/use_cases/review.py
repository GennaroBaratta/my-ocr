from __future__ import annotations

from my_ocr.domain import (
    LayoutDetectionFailed,
    MissingPage,
    OcrRuntimeOptions,
    ProviderArtifacts,
    ReviewLayout,
    ReviewPage,
    RunId,
    RunSnapshot,
    WorkflowResult,
)
from .ports import DocumentNormalizer, LayoutDetector, RunRepository


class ReviewUseCase:
    def __init__(
        self,
        run_store: RunRepository,
        normalizer: DocumentNormalizer,
        layout_detector: LayoutDetector,
    ) -> None:
        self._run_store = run_store
        self._normalizer = normalizer
        self._layout_detector = layout_detector

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


__all__ = ["ReviewUseCase"]
