from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_ocr_page(
    *,
    page_number: int = 1,
    page_path: str,
    markdown: str = "replacement markdown",
    markdown_source: str = "sdk_markdown",
    sdk_markdown: str = "",
    sdk_json_path: str,
    fallback_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "page_number": page_number,
        "page_path": page_path,
        "markdown": markdown,
        "markdown_source": markdown_source,
        "sdk_markdown": sdk_markdown,
        "sdk_json_path": sdk_json_path,
        "fallback_assessment": fallback_assessment or {},
    }


def build_fallback_chunk(
    *, crop_path: str, text_path: str, text: str = "fallback text"
) -> dict[str, Any]:
    return {
        "crop_path": crop_path,
        "text_path": text_path,
        "text": text,
    }


def build_fallback_page(
    *,
    page_number: int = 1,
    page_path: str,
    markdown: str = "replacement markdown",
    markdown_source: str = "crop_fallback",
    chunks: list[dict[str, Any]] | None = None,
    assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "page_number": page_number,
        "page_path": page_path,
        "assessment": assessment or {},
        "chunks": chunks or [],
        "markdown": markdown,
        "markdown_source": markdown_source,
    }


def write_normalized_page(run_dir: Path, *, content: bytes = b"replacement image") -> str:
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page = pages_dir / "page-0001.png"
    page.write_bytes(content)
    return str(page)


def normalize_to_single_page(_input_path: str, run_dir_arg: str | Path) -> list[str]:
    return [write_normalized_page(Path(run_dir_arg))]


def write_basic_ocr_outputs(
    run_dir: Path,
    *,
    markdown: str = "replacement markdown",
    json_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ocr.md").write_text(markdown, encoding="utf-8")
    payload = json_payload if json_payload is not None else {"replacement": True}
    (run_dir / "ocr.json").write_text(json.dumps(payload), encoding="utf-8")
    raw_dir = run_dir / "ocr_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return {
        "markdown": markdown,
        "json": payload,
        "raw_dir": str(raw_dir),
    }


def build_basic_ocr_result(
    run_dir_arg: str | Path,
    *,
    config_path: str,
    layout_device: str,
    markdown: str = "replacement markdown",
    json_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = write_basic_ocr_outputs(
        Path(run_dir_arg),
        markdown=markdown,
        json_payload=json_payload,
    )
    result.update({"config_path": config_path, "layout_device": layout_device})
    return result


def seed_existing_run(run_dir: Path) -> None:
    (run_dir / "pages").mkdir(parents=True, exist_ok=True)
    (run_dir / "pages" / "page-0001.png").write_bytes(b"existing page")
    raw_dir = run_dir / "ocr_raw" / "page-0001"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "page-0001_model.json").write_text(json.dumps({"legacy": True}), encoding="utf-8")
    (run_dir / "ocr.md").write_text("existing markdown", encoding="utf-8")
    (run_dir / "ocr.json").write_text('{"existing": true}', encoding="utf-8")

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    (predictions_dir / "glmocr_structured.json").write_text(
        json.dumps({"structured": "keep-me"}),
        encoding="utf-8",
    )
    (predictions_dir / "glmocr_structured_meta.json").write_text(
        json.dumps({"model": "keep-me"}),
        encoding="utf-8",
    )
    (predictions_dir / "rules.json").write_text(
        json.dumps({"rules": "existing"}),
        encoding="utf-8",
    )
    (predictions_dir / f"{run_dir.name}.json").write_text(
        json.dumps({"canonical": "keep-me"}),
        encoding="utf-8",
    )
