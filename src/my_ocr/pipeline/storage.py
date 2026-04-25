from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from my_ocr.application.artifacts import ProviderArtifacts
from my_ocr.models import (
    LayoutDiagnostics,
    OcrPageResult,
    OcrRunResult,
    PageRef,
    ReviewLayout,
    ReviewPage,
    RunId,
    RunSnapshot,
)


class RunHandle(Protocol):
    run_id: RunId
    work_dir: Path


class RunStorage(Protocol):
    def create_run(
        self, input_path: str | Path, run_id: RunId | None = None
    ) -> RunHandle: ...

    def open_run(self, run_id: RunId | str) -> RunSnapshot: ...

    def write_pages(self, run: RunHandle | RunId | str, pages: list[PageRef]) -> RunSnapshot | None: ...

    def write_review_layout(
        self,
        run: RunHandle | RunId | str,
        layout: ReviewLayout,
        artifacts: ProviderArtifacts,
        diagnostics: LayoutDiagnostics | None = None,
    ) -> RunSnapshot | None: ...

    def write_ocr_result(
        self,
        run_id: RunId | str,
        result: OcrRunResult,
        artifacts: ProviderArtifacts,
    ) -> RunSnapshot: ...

    def replace_page_layout(
        self,
        run_id: RunId | str,
        page_number: int,
        page: ReviewPage,
        artifacts: ProviderArtifacts,
    ) -> RunSnapshot: ...

    def replace_page_ocr(
        self,
        run_id: RunId | str,
        page_number: int,
        page: OcrPageResult,
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

    def clear_extraction_outputs(self, run: RunHandle | RunId | str) -> RunSnapshot | None: ...

    def commit_run(self, run: RunHandle) -> RunSnapshot: ...

    def rollback_run(self, run: RunHandle) -> None: ...

