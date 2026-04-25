from __future__ import annotations

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from my_ocr.support.filesystem import write_json


def save_result_to_raw_dir(
    result: Any,
    raw_dir: str | Path,
    page_path: str,
    page_number: int,
) -> Path:
    raw_root = Path(raw_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".page-{page_number:04d}-", dir=raw_root) as save_root:
        result.save(output_dir=save_root)
        return publish_saved_model_json_path(save_root, raw_root, page_path, page_number)


def write_page_layout_to_raw_dir(
    page_layout: dict[str, Any],
    raw_dir: str | Path,
    page_path: str,
    page_number: int,
) -> Path:
    target_dir = Path(raw_dir) / f"page-{page_number:04d}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / f"{Path(page_path).stem}_model.json"
    write_json(model_path, page_layout)
    return model_path


def publish_saved_model_json_path(
    save_root: str | Path,
    raw_root: str | Path,
    page_path: str,
    page_number: int,
) -> Path:
    page_stem = Path(page_path).stem
    source_dir = Path(save_root) / page_stem
    model_path = source_dir / f"{page_stem}_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved GLM-OCR model JSON: {model_path}")

    target_dir = Path(raw_root) / f"page-{page_number:04d}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(target_dir))
    return target_dir / model_path.name
