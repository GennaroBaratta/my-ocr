from __future__ import annotations

from dataclasses import dataclass

from my_ocr.application.dto import LayoutOptions, RunId, WorkflowResult
from my_ocr.application.errors import LayoutDetectionFailed, MissingPage
from my_ocr.application.ports import LayoutDetector, RunStore


@dataclass(frozen=True, slots=True)
class RerunPageLayoutCommand:
    run_id: RunId
    page_number: int
    options: LayoutOptions = LayoutOptions()


class RerunPageLayout:
    def __init__(self, run_store: RunStore, layout_detector: LayoutDetector) -> None:
        self._run_store = run_store
        self._layout_detector = layout_detector

    def __call__(self, command: RerunPageLayoutCommand) -> WorkflowResult:
        snapshot = self._run_store.open_run(command.run_id)
        page = snapshot.page(command.page_number)
        if page is None:
            raise MissingPage(f"Page {command.page_number} not found")
        try:
            result = self._layout_detector.detect_layout([page], command.options)
        except Exception as exc:
            raise LayoutDetectionFailed(f"Layout detection failed: {exc}") from exc
        if not result.layout.pages:
            raise MissingPage(f"Layout detector returned no page {command.page_number}")
        tx = self._run_store.begin_update(command.run_id)
        try:
            tx.replace_page_layout(command.page_number, result.layout.pages[0], result.artifacts)
            snapshot = tx.commit()
            return WorkflowResult(snapshot=snapshot, warning=result.diagnostics.warning)
        except Exception:
            tx.rollback()
            raise

