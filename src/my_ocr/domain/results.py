from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

from pydantic import Field

from my_ocr.domain._base import StrictModel
from my_ocr.domain.artifacts import ProviderArtifacts
from my_ocr.domain.ocr import OcrRunResult
from my_ocr.domain.review import ReviewLayout
from my_ocr.domain.run import LayoutDiagnostics, PageRef, RunId, RunManifest


class RunSnapshot(StrictModel):
    run_dir: Path
    manifest: RunManifest
    review_layout: ReviewLayout | None = None
    ocr_result: OcrRunResult | None = None
    extraction: dict[str, Any] = Field(default_factory=dict)

    @property
    def run_id(self) -> RunId:
        return self.manifest.run_id

    @property
    def pages(self) -> list[PageRef]:
        return self.manifest.pages

    def page(self, page_number: int) -> PageRef | None:
        for page in self.manifest.pages:
            if page.page_number == page_number:
                return page
        return None


@dataclass(frozen=True, slots=True)
class LayoutDetectionResult:
    layout: ReviewLayout
    artifacts: ProviderArtifacts = dataclass_field(default_factory=ProviderArtifacts.empty)
    diagnostics: LayoutDiagnostics = dataclass_field(default_factory=LayoutDiagnostics)


@dataclass(frozen=True, slots=True)
class OcrRecognitionResult:
    result: OcrRunResult
    artifacts: ProviderArtifacts = dataclass_field(default_factory=ProviderArtifacts.empty)


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    snapshot: RunSnapshot
    warning: str | None = None
