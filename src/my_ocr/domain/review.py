from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from my_ocr.domain._base import StrictModel
from my_ocr.domain._base import validate_optional_run_relative_path, validate_run_relative_path


class LayoutBlock(StrictModel):
    id: str
    index: int
    label: str
    bbox: list[float]
    confidence: float = 1.0
    content: str = ""


class ReviewPage(StrictModel):
    page_number: int
    image_path: str
    image_width: int
    image_height: int
    blocks: list[LayoutBlock]
    provider_path: str | None = None
    coord_space: str = "pixel"

    @field_validator("image_path")
    @classmethod
    def _validate_image_path(cls, value: str) -> str:
        return validate_run_relative_path(value, field_name="image_path")

    @field_validator("provider_path")
    @classmethod
    def _validate_provider_path(cls, value: str | None) -> str | None:
        return validate_optional_run_relative_path(value, field_name="provider_path")


class ReviewLayout(StrictModel):
    pages: list[ReviewPage]
    status: Literal["prepared", "reviewed"] = "prepared"
    version: int = 3

    @property
    def summary(self) -> dict[str, int]:
        return {"page_count": len(self.pages)}
