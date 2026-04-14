from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .evaluate import evaluate_directories, write_markdown_report
from .extract_glmocr import extract_structured, save_structured_result
from .extract_rules import extract_from_markdown
from .ingest import normalize_document
from .ocr import run_ocr
from .paths import RunPaths
from .settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LAYOUT_DEVICE,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RUN_ROOT,
)
from .utils import write_json


def run_ocr_workflow(
    input_path: str,
    *,
    run: str | None = None,
    run_root: str = DEFAULT_RUN_ROOT,
    config_path: str = DEFAULT_CONFIG_PATH,
    layout_device: str = DEFAULT_LAYOUT_DEVICE,
    normalize_document_fn: Callable[[str, str | Path], list[str]] = normalize_document,
    run_ocr_fn: Callable[..., dict[str, Any]] = run_ocr,
    write_json_fn: Callable[[str | Path, Any], None] = write_json,
) -> Path:
    paths = RunPaths.from_input(input_path, run=run, run_root=run_root)
    paths.ensure_run_dir()
    staged_paths = _create_staged_ocr_paths(paths)
    try:
        pages = normalize_document_fn(input_path, staged_paths.run_dir)
        result = run_ocr_fn(
            pages,
            staged_paths.run_dir,
            config_path=config_path,
            layout_device=layout_device,
        )
        _publish_staged_ocr_run(staged_paths, paths)
    finally:
        shutil.rmtree(staged_paths.run_dir, ignore_errors=True)

    final_pages = paths.list_page_paths()
    write_run_metadata(
        paths.run_dir,
        input_path,
        final_pages,
        {
            **result,
            "raw_dir": str(paths.raw_dir),
        },
        write_json_fn=write_json_fn,
    )
    return paths.run_dir


def run_rules_workflow(
    run: str,
    *,
    run_root: str = DEFAULT_RUN_ROOT,
    extract_from_markdown_fn: Callable[[str], dict[str, Any]] = extract_from_markdown,
    write_json_fn: Callable[[str | Path, Any], None] = write_json,
) -> None:
    paths = RunPaths.from_named_run(run, run_root=run_root)
    if not paths.ocr_markdown_path.exists():
        raise FileNotFoundError(f"Missing OCR markdown: {paths.ocr_markdown_path}")

    prediction = extract_from_markdown_fn(paths.ocr_markdown_path.read_text(encoding="utf-8"))
    write_json_fn(paths.rules_prediction_path, prediction)
    write_json_fn(paths.canonical_prediction_path, prediction)


def run_structured_workflow(
    run: str,
    *,
    run_root: str = DEFAULT_RUN_ROOT,
    model: str = DEFAULT_OLLAMA_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    extract_structured_fn: Callable[
        ..., tuple[dict[str, Any], dict[str, Any]]
    ] = extract_structured,
    save_structured_result_fn: Callable[
        [str | Path, dict[str, Any], dict[str, Any]], None
    ] = save_structured_result,
    write_json_fn: Callable[[str | Path, Any], None] = write_json,
) -> None:
    paths = RunPaths.from_named_run(run, run_root=run_root)
    page_paths = paths.list_page_paths()
    if not page_paths:
        raise FileNotFoundError(f"No page images found in {paths.pages_dir}")

    prediction, metadata = extract_structured_fn(page_paths, model=model, endpoint=endpoint)
    save_structured_result_fn(paths.run_dir, prediction, metadata)
    write_json_fn(paths.canonical_prediction_path, prediction)


def run_pipeline_workflow(
    input_path: str,
    *,
    run: str | None = None,
    run_root: str = DEFAULT_RUN_ROOT,
    config_path: str = DEFAULT_CONFIG_PATH,
    layout_device: str = DEFAULT_LAYOUT_DEVICE,
    normalize_document_fn: Callable[[str, str | Path], list[str]] = normalize_document,
    run_ocr_fn: Callable[..., dict[str, Any]] = run_ocr,
    extract_from_markdown_fn: Callable[[str], dict[str, Any]] = extract_from_markdown,
    write_json_fn: Callable[[str | Path, Any], None] = write_json,
) -> Path:
    run_dir = run_ocr_workflow(
        input_path,
        run=run,
        run_root=run_root,
        config_path=config_path,
        layout_device=layout_device,
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


def evaluate_workflow(
    gold_dir: str,
    pred_dir: str,
    output: str,
    *,
    evaluate_directories_fn: Callable[
        [str | Path, str | Path], dict[str, Any]
    ] = evaluate_directories,
    write_markdown_report_fn: Callable[[dict[str, Any], str | Path], None] = write_markdown_report,
) -> dict[str, Any]:
    report = evaluate_directories_fn(gold_dir, pred_dir)
    write_markdown_report_fn(report, output)
    return report


def write_run_metadata(
    run_dir: str | Path,
    input_path: str,
    page_paths: list[str],
    ocr_result: dict[str, Any],
    *,
    write_json_fn: Callable[[str | Path, Any], None] = write_json,
) -> None:
    paths = RunPaths.from_run_dir(run_dir)
    payload = {
        "input_path": str(input_path),
        "page_paths": page_paths,
        "ocr_raw_dir": ocr_result.get("raw_dir"),
        "fallback_used": bool(ocr_result.get("fallback_used")),
    }
    write_json_fn(paths.meta_path, payload)


def _create_staged_ocr_paths(paths: RunPaths) -> RunPaths:
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{paths.run_name}-ocr-", dir=str(paths.run_dir.parent))
    )
    return RunPaths.from_run_dir(staging_dir)


def _publish_staged_ocr_run(source_paths: RunPaths, target_paths: RunPaths) -> None:
    _replace_dir(source_paths.pages_dir, target_paths.pages_dir)
    _replace_dir(source_paths.raw_dir, target_paths.raw_dir)
    _replace_dir(source_paths.fallback_dir, target_paths.fallback_dir)
    _replace_file(source_paths.ocr_markdown_path, target_paths.ocr_markdown_path)
    _replace_file(source_paths.ocr_json_path, target_paths.ocr_json_path)
    _replace_file(source_paths.ocr_fallback_path, target_paths.ocr_fallback_path)


def _replace_dir(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        target.mkdir(parents=True, exist_ok=True)
        return
    source.replace(target)


def _replace_file(source: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
