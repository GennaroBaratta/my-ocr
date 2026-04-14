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
    def rules_prediction_path(self) -> Path:
        return self.predictions_dir / "rules.json"

    @property
    def structured_prediction_path(self) -> Path:
        return self.predictions_dir / "glmocr_structured.json"

    @property
    def structured_metadata_path(self) -> Path:
        return self.predictions_dir / "glmocr_structured_meta.json"

    @property
    def canonical_prediction_path(self) -> Path:
        return self.predictions_dir / f"{self.run_name}.json"

    def ensure_run_dir(self) -> Path:
        return ensure_dir(self.run_dir)

    def reset_pages_dir(self) -> Path:
        return _reset_dir(self.pages_dir)

    def reset_predictions_dir(self) -> Path:
        return _reset_dir(self.predictions_dir)

    def reset_ocr_artifacts(self) -> None:
        _reset_dir(self.raw_dir)
        _reset_dir(self.fallback_dir)
        _remove_file(self.ocr_markdown_path)
        _remove_file(self.ocr_json_path)
        _remove_file(self.ocr_fallback_path)
        _remove_file(self.meta_path)

    def reset_for_ocr_run(self) -> None:
        self.reset_pages_dir()
        self.reset_ocr_artifacts()

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
