from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from my_ocr.application.ports import JsonWriter, StructuredExtractor
from my_ocr.application.services.structured_validation import validate_structured_prediction

from ._run_state import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RUN_ROOT,
    RunPaths,
    clear_structured_outputs,
    load_json,
    write_json,
)


def run_structured_workflow(
    run: str,
    *,
    run_root: str = DEFAULT_RUN_ROOT,
    config_path: str = DEFAULT_CONFIG_PATH,
    model: str | None = None,
    endpoint: str | None = None,
    extract_structured_fn: StructuredExtractor | None = None,
    save_structured_result_fn: Callable[[str | Path, dict[str, Any], dict[str, Any]], None]
    | None = None,
    write_json_fn: JsonWriter = write_json,
) -> None:
    if extract_structured_fn is None:
        structured_adapter = import_module("my_ocr.adapters.outbound.llm.structured_extractor")
        extract_structured_fn = structured_adapter.extract_structured
    if save_structured_result_fn is None:
        structured_adapter = import_module("my_ocr.adapters.outbound.llm.structured_extractor")
        save_structured_result_fn = structured_adapter.save_structured_result

    paths = RunPaths.from_named_run(run, run_root=run_root)
    page_paths = paths.list_page_paths()
    if not page_paths:
        raise FileNotFoundError(f"No page images found in {paths.pages_dir}")

    markdown_text = None
    if paths.ocr_markdown_path.exists():
        markdown_text = paths.ocr_markdown_path.read_text(encoding="utf-8")

    try:
        prediction, metadata = extract_structured_fn(
            page_paths,
            markdown_text=markdown_text,
            config_path=config_path,
            model=model,
            endpoint=endpoint,
        )
    except RuntimeError as exc:
        if not paths.rules_prediction_path.exists():
            raise RuntimeError(
                "Structured extraction failed and no rules prediction is available. "
                "Run 'extract-rules' or 'run' first, then retry 'extract-glmocr'."
            ) from exc

        rules_prediction = load_json(paths.rules_prediction_path)
        write_json_fn(paths.canonical_prediction_path, rules_prediction)
        write_json_fn(
            paths.structured_metadata_path,
            {
                "status": "failed",
                "error": str(exc),
                "canonical_source": "rules",
            },
        )
        clear_structured_outputs(paths, keep_metadata=True)
        return
    validation = validate_structured_prediction(prediction, source_text=markdown_text)
    canonical_prediction = prediction
    canonical_source = "glmocr_structured"
    if not validation["ok"] and paths.rules_prediction_path.exists():
        canonical_prediction = load_json(paths.rules_prediction_path)
        canonical_source = "rules"

    metadata = {
        **metadata,
        "canonical_source": canonical_source,
        "validation": validation,
    }
    save_structured_result_fn(paths.run_dir, prediction, metadata)
    write_json_fn(paths.canonical_prediction_path, canonical_prediction)


__all__ = ["run_structured_workflow"]
