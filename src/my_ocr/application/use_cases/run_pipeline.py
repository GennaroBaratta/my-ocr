from __future__ import annotations

from pathlib import Path

from my_ocr.application.ports import DocumentNormalizer, JsonWriter, MarkdownExtractor, OcrEngine
from my_ocr.application.services.rules_extractor import extract_from_markdown

from ._run_state import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LAYOUT_DEVICE,
    DEFAULT_RUN_ROOT,
    normalize_document,
    run_ocr,
    write_json,
)
from .extract_rules import run_rules_workflow
from .run_ocr import run_ocr_workflow


def run_pipeline_workflow(
    input_path: str,
    *,
    run: str | None = None,
    run_root: str = DEFAULT_RUN_ROOT,
    config_path: str = DEFAULT_CONFIG_PATH,
    layout_device: str = DEFAULT_LAYOUT_DEVICE,
    layout_profile: str | None = "auto",
    normalize_document_fn: DocumentNormalizer = normalize_document,
    run_ocr_fn: OcrEngine = run_ocr,
    extract_from_markdown_fn: MarkdownExtractor = extract_from_markdown,
    write_json_fn: JsonWriter = write_json,
) -> Path:
    run_dir = run_ocr_workflow(
        input_path,
        run=run,
        run_root=run_root,
        config_path=config_path,
        layout_device=layout_device,
        layout_profile=layout_profile,
        normalize_document_fn=normalize_document_fn,
        run_ocr_fn=run_ocr_fn,
        write_json_fn=write_json_fn,
    )
    run_rules_workflow(
        run_dir.name,
        run_root=run_root,
        extract_from_markdown_fn=extract_from_markdown_fn,
        write_json_fn=write_json_fn,
    )
    return run_dir


__all__ = ["run_pipeline_workflow"]
