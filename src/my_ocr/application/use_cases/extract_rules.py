from __future__ import annotations

from my_ocr.application.ports import JsonWriter, MarkdownExtractor
from my_ocr.application.services.rules_extractor import extract_from_markdown

from ._run_state import DEFAULT_RUN_ROOT, RunPaths, write_json


def run_rules_workflow(
    run: str,
    *,
    run_root: str = DEFAULT_RUN_ROOT,
    extract_from_markdown_fn: MarkdownExtractor = extract_from_markdown,
    write_json_fn: JsonWriter = write_json,
) -> None:
    paths = RunPaths.from_named_run(run, run_root=run_root)
    if not paths.ocr_markdown_path.exists():
        raise FileNotFoundError(f"Missing OCR markdown: {paths.ocr_markdown_path}")

    prediction = extract_from_markdown_fn(paths.ocr_markdown_path.read_text(encoding="utf-8"))
    write_json_fn(paths.rules_prediction_path, prediction)
    write_json_fn(paths.canonical_prediction_path, prediction)


__all__ = ["run_rules_workflow"]
