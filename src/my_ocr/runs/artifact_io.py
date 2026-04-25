from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from my_ocr.support.filesystem import read_json, read_text, write_json, write_text
from my_ocr.domain import OcrRunResult, ReviewLayout


@dataclass(frozen=True, slots=True)
class RunLayoutPaths:
    run_dir: Path

    @property
    def manifest(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def pages_dir(self) -> Path:
        return self.run_dir / "pages"

    @property
    def layout_dir(self) -> Path:
        return self.run_dir / "layout"

    @property
    def review_layout(self) -> Path:
        return self.layout_dir / "review.json"

    @property
    def ocr_dir(self) -> Path:
        return self.run_dir / "ocr"

    @property
    def ocr_markdown(self) -> Path:
        return self.ocr_dir / "markdown.md"

    @property
    def ocr_pages(self) -> Path:
        return self.ocr_dir / "pages.json"

    @property
    def extraction_dir(self) -> Path:
        return self.run_dir / "extraction"

    @property
    def rules_extraction(self) -> Path:
        return self.extraction_dir / "rules.json"

    @property
    def structured_extraction(self) -> Path:
        return self.extraction_dir / "structured.json"

    @property
    def structured_extraction_meta(self) -> Path:
        return self.extraction_dir / "structured_meta.json"

    @property
    def structured_extraction_raw(self) -> Path:
        return self.extraction_dir / "structured_raw.json"

    @property
    def canonical_extraction(self) -> Path:
        return self.extraction_dir / "canonical.json"


def write_review_layout_payload(run_dir: Path, layout: ReviewLayout) -> None:
    write_json(RunLayoutPaths(run_dir).review_layout, layout.model_dump(mode="json"))


def write_ocr_result_payload(run_dir: Path, result: OcrRunResult) -> None:
    paths = RunLayoutPaths(run_dir)
    write_text(paths.ocr_markdown, result.markdown)
    write_json(paths.ocr_pages, result.model_dump(mode="json"))


def load_review_layout(run_dir: Path) -> ReviewLayout | None:
    path = RunLayoutPaths(run_dir).review_layout
    if not path.exists():
        return None
    return ReviewLayout.model_validate(read_json(path))


def load_ocr_result(run_dir: Path) -> OcrRunResult | None:
    paths = RunLayoutPaths(run_dir)
    if not paths.ocr_pages.exists():
        return None
    payload = read_json(paths.ocr_pages)
    if not isinstance(payload, dict):
        return None
    if "markdown" not in payload and paths.ocr_markdown.exists():
        payload = {**payload, "markdown": read_text(paths.ocr_markdown)}
    return OcrRunResult.model_validate(payload)


def load_extraction(run_dir: Path) -> dict[str, Any]:
    paths = RunLayoutPaths(run_dir)
    if not paths.extraction_dir.exists():
        return {}
    payload: dict[str, Any] = {}
    for key, path in (
        ("rules", paths.rules_extraction),
        ("structured", paths.structured_extraction),
        ("structured_meta", paths.structured_extraction_meta),
        ("structured_raw", paths.structured_extraction_raw),
        ("canonical", paths.canonical_extraction),
    ):
        if path.exists():
            payload[key] = read_json(path)
    return payload
