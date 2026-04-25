from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from my_ocr.domain._base import StrictModel
from my_ocr.domain._base import validate_optional_run_relative_path, validate_run_relative_path


class OcrPageResult(StrictModel):
    page_number: int
    image_path: str
    markdown: str
    markdown_source: str = "unknown"
    provider_path: str | None = None
    fallback_path: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image_path")
    @classmethod
    def _validate_image_path(cls, value: str) -> str:
        return validate_run_relative_path(value, field_name="image_path")

    @field_validator("provider_path", "fallback_path")
    @classmethod
    def _validate_artifact_path(cls, value: str | None) -> str | None:
        return validate_optional_run_relative_path(value, field_name="artifact_path")


class OcrRunResult(StrictModel):
    pages: list[OcrPageResult]
    markdown: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        return {"page_count": len(self.pages), "sources": self.source_counts()}

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for page in self.pages:
            counts[page.markdown_source] = counts.get(page.markdown_source, 0) + 1
        return counts
