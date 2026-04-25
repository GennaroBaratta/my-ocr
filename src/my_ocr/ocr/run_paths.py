from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Temporary provider artifact paths for one GLM-OCR run."""

    run_dir: Path

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "RunPaths":
        return cls(Path(run_dir))

    @property
    def raw_dir(self) -> Path:
        return self.run_dir / "ocr_raw"

    @property
    def fallback_dir(self) -> Path:
        return self.run_dir / "ocr_fallback"

    def ensure_run_dir(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def reset_ocr_artifacts(self) -> None:
        for path in (
            self.raw_dir,
            self.fallback_dir,
        ):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fallback_page_dir(self, page_number: int) -> Path:
        path = self.fallback_dir / f"page-{page_number:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

