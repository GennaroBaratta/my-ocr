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
        if len(page_paths) == 1:
            results = [parser.parse(page_paths[0])]
        else:
            parse_inputs: list[str | bytes | Path] = list(page_paths)
            results = parser.parse(parse_inputs)

        markdown_parts: list[str] = []
        json_parts: list[Any] = []
        for index, result in enumerate(results, start=1):
            page_raw_dir = ensure_dir(raw_dir / f"page-{index:04d}")
            result.save(output_dir=str(page_raw_dir))
            markdown_parts.append(getattr(result, "markdown_result", "") or "")
            json_parts.append(getattr(result, "json_result", {}) or {})

        markdown = "\n\n---\n\n".join(part for part in markdown_parts if part)
        json_result = json_parts[0] if len(json_parts) == 1 else json_parts

    write_text(run_dir / "ocr.md", markdown)
    write_json(run_dir / "ocr.json", json_result)

    return {
        "markdown": markdown,
        "json": json_result,
        "raw_dir": str(raw_dir),
    }
