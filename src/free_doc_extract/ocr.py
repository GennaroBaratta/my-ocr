from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import ocr_fallback as _ocr_fallback
from .paths import RunPaths
from .utils import write_json, write_text

_build_text_chunks = _ocr_fallback.build_text_chunks
_clean_recognized_text = _ocr_fallback.clean_recognized_text
_needs_crop_fallback = _ocr_fallback.needs_crop_fallback
_run_crop_fallback_for_page = _ocr_fallback.run_crop_fallback_for_page


@dataclass(slots=True)
class ProcessedOcrPage:
    markdown: str
    json_result: Any
    fallback_page: dict[str, Any] | None = None


def run_ocr(
    page_paths: list[str],
    run_dir: str | Path,
    *,
    config_path: str = "config/local.yaml",
    layout_device: str = "cpu",
) -> dict[str, Any]:
    if not page_paths:
        raise ValueError("page_paths cannot be empty")

    parser_cls = _load_glmocr_parser()
    paths = RunPaths.from_run_dir(run_dir)
    paths.ensure_run_dir()
    paths.reset_ocr_artifacts()

    with parser_cls(config_path=config_path, layout_device=layout_device) as parser:
        processed_pages = [
            _process_page_result(
                page_number=index,
                page_path=page_paths[index - 1],
                result=result,
                paths=paths,
            )
            for index, result in enumerate(_parse_pages(parser, page_paths), start=1)
        ]

    markdown = _join_markdown_pages(page.markdown for page in processed_pages)
    json_result = _combine_json_pages([page.json_result for page in processed_pages])
    fallback_pages = [
        page.fallback_page for page in processed_pages if page.fallback_page is not None
    ]

    write_text(paths.ocr_markdown_path, markdown)
    write_json(paths.ocr_json_path, json_result)
    if fallback_pages:
        write_json(paths.ocr_fallback_path, fallback_pages)

    return {
        "markdown": markdown,
        "json": json_result,
        "raw_dir": str(paths.raw_dir),
        "fallback_used": bool(fallback_pages),
    }


def _load_glmocr_parser() -> type:
    try:
        from glmocr import GlmOcr
    except ImportError as exc:
        raise RuntimeError(
            "GLM-OCR is not installed. Install with `pip install -e .[glmocr]`."
        ) from exc
    return GlmOcr


def _parse_pages(parser: Any, page_paths: list[str]) -> list[Any]:
    if len(page_paths) == 1:
        return [parser.parse(page_paths[0])]

    parse_inputs: list[str | bytes | Path] = list(page_paths)
    return list(parser.parse(parse_inputs))


def _process_page_result(
    *,
    page_number: int,
    page_path: str,
    result: Any,
    paths: RunPaths,
) -> ProcessedOcrPage:
    result.save(output_dir=str(paths.raw_page_dir(page_number)))

    page_markdown = getattr(result, "markdown_result", "") or ""
    page_json = getattr(result, "json_result", {}) or {}
    page_layout_json = getattr(result, "raw_json_result", None) or page_json
    effective_markdown, fallback_page = _maybe_apply_crop_fallback(
        page_number=page_number,
        page_path=page_path,
        page_markdown=page_markdown,
        page_layout_json=page_layout_json,
        paths=paths,
    )
    return ProcessedOcrPage(
        markdown=effective_markdown,
        json_result=page_json,
        fallback_page=fallback_page,
    )


def _maybe_apply_crop_fallback(
    *,
    page_number: int,
    page_path: str,
    page_markdown: str,
    page_layout_json: Any,
    paths: RunPaths,
) -> tuple[str, dict[str, Any] | None]:
    if not _needs_crop_fallback(page_markdown, page_layout_json):
        return page_markdown, None

    try:
        fallback_markdown, fallback_meta = _run_crop_fallback_for_page(
            page_path=page_path,
            page_json=page_layout_json,
            page_fallback_dir=paths.fallback_page_dir(page_number),
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        fallback_markdown = ""
        fallback_meta = [{"chunk": 0, "text": "", "error": str(exc)}]

    fallback_page = {
        "page": page_number,
        "page_path": page_path,
        "recovered_text": bool(fallback_markdown),
        "chunks": fallback_meta,
    }
    return fallback_markdown or page_markdown, fallback_page


def _join_markdown_pages(markdown_parts: Iterable[str]) -> str:
    return "\n\n---\n\n".join(part for part in markdown_parts if part)


def _combine_json_pages(json_parts: list[Any]) -> Any:
    return json_parts[0] if len(json_parts) == 1 else json_parts
