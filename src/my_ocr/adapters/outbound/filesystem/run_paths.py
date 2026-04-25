from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Adapter-internal legacy layout used by the GLM-OCR bridge in temp folders."""

    run_dir: Path

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "RunPaths":
        return cls(Path(run_dir))

    @property
    def run_name(self) -> str:
        return self.run_dir.name

    @property
    def pages_dir(self) -> Path:
        return self.run_dir / "pages"

    @property
    def raw_dir(self) -> Path:
        return self.run_dir / "ocr_raw"

    @property
    def fallback_dir(self) -> Path:
        return self.run_dir / "ocr_fallback"

    @property
    def ocr_markdown_path(self) -> Path:
        return self.run_dir / "ocr.md"

    @property
    def ocr_json_path(self) -> Path:
        return self.run_dir / "ocr.json"

    @property
    def ocr_fallback_path(self) -> Path:
        return self.run_dir / "ocr_fallback.json"

    @property
    def reviewed_layout_path(self) -> Path:
        return self.run_dir / "reviewed_layout.json"

    def ensure_run_dir(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def reset_ocr_artifacts(self) -> None:
        for path in (
            self.raw_dir,
            self.fallback_dir,
            self.ocr_markdown_path,
            self.ocr_json_path,
            self.ocr_fallback_path,
        ):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def raw_page_dir(self, page_number: int) -> Path:
        path = self.raw_dir / f"page-{page_number:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fallback_page_dir(self, page_number: int) -> Path:
        path = self.fallback_dir / f"page-{page_number:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

