from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_json, write_text


def run_ocr(
    page_paths: list[str],
    run_dir: str | Path,
    *,
    config_path: str = "config/local.yaml",
    layout_device: str = "cpu",
) -> dict[str, Any]:
    if not page_paths:
        raise ValueError("page_paths cannot be empty")

    try:
        from glmocr import GlmOcr
    except ImportError as exc:
        raise RuntimeError(
            "GLM-OCR is not installed. Install with `pip install -e .[glmocr]`."
        ) from exc

    run_dir = ensure_dir(run_dir)
    raw_dir = ensure_dir(run_dir / "ocr_raw")

    with GlmOcr(config_path=config_path, layout_device=layout_device) as parser:
        result = parser.parse(page_paths if len(page_paths) > 1 else page_paths[0])
        result.save(output_dir=str(raw_dir))

        markdown = getattr(result, "markdown_result", "") or ""
        json_result = getattr(result, "json_result", {}) or {}

    write_text(run_dir / "ocr.md", markdown)
    write_json(run_dir / "ocr.json", json_result)

    return {
        "markdown": markdown,
        "json": json_result,
        "raw_dir": str(raw_dir),
    }
