from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from my_ocr.application.artifacts import (
    LayoutDetectionResult,
    OcrRecognitionResult,
    ProviderArtifacts,
)
from my_ocr.application.models import (
    LayoutDiagnostics,
    OcrPageResult,
    OcrRunResult,
    PageRef,
    ReviewLayout,
    ReviewPage,
    RunId,
    RunSnapshot,
)
from my_ocr.application.options import (
    LayoutOptions,
    OcrOptions,
    StructuredExtractionOptions,
)


class RunTransaction(Protocol):
    run_id: RunId
    work_dir: Path

    def write_pages(self, pages: list[PageRef]) -> None: ...

    def write_review_layout(
        self,
        layout: ReviewLayout,
        artifacts: ProviderArtifacts,
        diagnostics: LayoutDiagnostics | None = None,
    ) -> None: ...

    def write_ocr_result(
        self, result: OcrRunResult, artifacts: ProviderArtifacts
    ) -> None: ...

    def replace_page_layout(
        self, page_number: int, page: ReviewPage, artifacts: ProviderArtifacts
    ) -> None: ...

    def replace_page_ocr(
        self, page_number: int, page: OcrPageResult, artifacts: ProviderArtifacts
    ) -> None: ...

    def write_rules_extraction(self, prediction: dict[str, Any]) -> None: ...

    def write_structured_extraction(
        self,
        prediction: dict[str, Any],
        metadata: dict[str, Any],
        *,
        canonical_prediction: dict[str, Any],
    ) -> None: ...

    def clear_extraction_outputs(self) -> None: ...

    def commit(self) -> RunSnapshot: ...

    def rollback(self) -> None: ...


class RunStore(Protocol):
    def create_run(self, input_path: str | Path, run_id: RunId | None = None) -> RunTransaction: ...

    def open_run(self, run_id: RunId | str) -> RunSnapshot: ...

    def begin_update(self, run_id: RunId | str) -> RunTransaction: ...


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
