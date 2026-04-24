from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .settings import DEFAULT_RUN_ROOT
from .utils import ensure_dir, slugify, timestamp_id


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_dir: Path

    @classmethod
    def from_input(
        cls,
        input_path: str | Path,
        *,
        run: str | None = None,
        run_root: str = DEFAULT_RUN_ROOT,
    ) -> "RunPaths":
        source = Path(input_path)
        run_id = run or f"{slugify(source.stem)}-{timestamp_id()}"
        return cls(Path(run_root) / run_id)

    @classmethod
    def from_named_run(cls, run: str, *, run_root: str = DEFAULT_RUN_ROOT) -> "RunPaths":
        run_dir = Path(run_root) / run
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return cls(run_dir)

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
    def predictions_dir(self) -> Path:
        return self.run_dir / "predictions"

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
    def meta_path(self) -> Path:
        return self.run_dir / "meta.json"

    @property
    def reviewed_layout_path(self) -> Path:
        return self.run_dir / "reviewed_layout.json"

    @property
    def rules_prediction_path(self) -> Path:
        return self.predictions_dir / "rules.json"

    @property
    def structured_prediction_path(self) -> Path:
        return self.predictions_dir / "glmocr_structured.json"

    @property
    def structured_metadata_path(self) -> Path:
        return self.predictions_dir / "glmocr_structured_meta.json"

    @property
    def structured_raw_path(self) -> Path:
        return self.predictions_dir / "glmocr_structured_raw.json"

    @property
    def canonical_prediction_path(self) -> Path:
        return self.predictions_dir / f"{self.run_name}.json"

    def ensure_run_dir(self) -> Path:
        return ensure_dir(self.run_dir)

    def _ocr_output_paths(self) -> tuple[Path, ...]:
        return (
            self.raw_dir,
            self.fallback_dir,
            self.ocr_markdown_path,
            self.ocr_json_path,
            self.ocr_fallback_path,
        )

    def published_ocr_artifact_paths(self) -> tuple[Path, ...]:
        return (self.pages_dir, *self._ocr_output_paths())

    def published_reviewed_ocr_artifact_paths(self) -> tuple[Path, ...]:
        return self._ocr_output_paths()

    def resettable_ocr_artifact_paths(self) -> tuple[Path, ...]:
        return (*self._ocr_output_paths(), self.meta_path)

    def published_review_artifact_paths(self) -> tuple[Path, ...]:
        return (self.pages_dir, self.raw_dir, self.reviewed_layout_path)

    def reset_ocr_artifacts(self) -> None:
        for path in self.resettable_ocr_artifact_paths():
            if path == self.raw_dir:
                _reset_dir(path)
            elif path == self.fallback_dir:
                _remove_dir(path)
            else:
                _remove_file(path)

    def raw_page_dir(self, page_number: int) -> Path:
        return ensure_dir(self.raw_dir / f"page-{page_number:04d}")

    def fallback_page_dir(self, page_number: int) -> Path:
        return ensure_dir(self.fallback_dir / f"page-{page_number:04d}")

    def list_page_paths(self) -> list[str]:
        if not self.pages_dir.exists():
            return []
        return sorted(str(path) for path in self.pages_dir.iterdir() if path.is_file())


def _reset_dir(path: str | Path) -> Path:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _remove_file(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()


def _remove_dir(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)
