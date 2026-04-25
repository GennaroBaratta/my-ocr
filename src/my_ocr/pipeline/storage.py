from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

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
from my_ocr.pipeline.types import ProviderArtifacts


class RunWorkspace(Protocol):
    run_id: RunId
    work_dir: Path


class RunStorage(Protocol):
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

    def write_review_layout(
        self,
        run_id: RunId | str,
        layout: ReviewLayout,
        artifacts: ProviderArtifacts,
        diagnostics: LayoutDiagnostics | None = None,
    ) -> RunSnapshot: ...

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

    def clear_extraction_outputs(self, run_id: RunId | str) -> RunSnapshot: ...
