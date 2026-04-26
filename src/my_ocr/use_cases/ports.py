from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from my_ocr.domain import (
    LayoutDetectionResult,
    LayoutDiagnostics,
    OcrRecognitionResult,
    OcrRunResult,
    OcrRuntimeOptions,
    PageRef,
    ProviderArtifacts,
    ReviewLayout,
    RunId,
    RunSnapshot,
    StructuredExtractionOptions,
)


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


__all__ = [
    "DocumentNormalizer",
    "LayoutDetector",
    "OcrEngine",
    "RulesExtractor",
    "RunRepository",
    "RunWorkspace",
    "StructuredExtractor",
]
