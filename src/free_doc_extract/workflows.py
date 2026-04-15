from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .evaluate import evaluate_directories, write_markdown_report
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
from .utils import collapse_whitespace, load_json, write_json


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
        page_paths = normalize_document_fn(input_path, staged_paths.run_dir)
        result = run_ocr_fn(
            page_paths,
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
    config_path: str = DEFAULT_CONFIG_PATH,
    model: str | None = None,
    endpoint: str | None = None,
    extract_structured_fn: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None = None,
    save_structured_result_fn: Callable[[str | Path, dict[str, Any], dict[str, Any]], None]
    | None = None,
    write_json_fn: Callable[[str | Path, Any], None] = write_json,
) -> None:
    if extract_structured_fn is None:
        from .experimental.extract_glmocr import extract_structured

        extract_structured_fn = extract_structured
    if save_structured_result_fn is None:
        from .experimental.extract_glmocr import save_structured_result

        save_structured_result_fn = save_structured_result

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
        _clear_structured_outputs(paths, keep_metadata=True)
        return
    validation = _validate_structured_prediction(prediction)
    if markdown_text:
        validation = _validate_structured_prediction(prediction, source_text=markdown_text)
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
        "config_path": ocr_result.get("config_path"),
        "layout_device": ocr_result.get("layout_device"),
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
    _replace_file(source_paths.ocr_markdown_path, target_paths.ocr_markdown_path)
    _replace_file(source_paths.ocr_json_path, target_paths.ocr_json_path)
    _replace_optional_dir(source_paths.fallback_dir, target_paths.fallback_dir)
    _replace_file(source_paths.ocr_fallback_path, target_paths.ocr_fallback_path)
    _rewrite_published_run_paths(
        target_paths.ocr_json_path, source_paths.run_dir, target_paths.run_dir
    )
    _rewrite_published_run_paths(
        target_paths.ocr_fallback_path, source_paths.run_dir, target_paths.run_dir
    )


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


def _replace_optional_dir(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)


def _rewrite_published_run_paths(
    payload_path: Path, source_run_dir: str | Path, target_run_dir: str | Path
) -> None:
    if not payload_path.exists():
        return

    source_prefix = str(source_run_dir)
    target_prefix = str(target_run_dir)
    payload = load_json(payload_path)
    rewritten = _rewrite_path_strings(payload, source_prefix, target_prefix)
    write_json(payload_path, rewritten)


def _rewrite_path_strings(
    payload: Any,
    source_prefix: str,
    target_prefix: str,
    *,
    parent_key: str | None = None,
) -> Any:
    if isinstance(payload, dict):
        return {
            key: _rewrite_path_strings(value, source_prefix, target_prefix, parent_key=key)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [
            _rewrite_path_strings(value, source_prefix, target_prefix, parent_key=parent_key)
            for value in payload
        ]
    if isinstance(payload, str) and source_prefix in payload and _is_path_like_key(parent_key):
        return payload.replace(source_prefix, target_prefix)
    return payload


def _is_path_like_key(key: str | None) -> bool:
    if key is None:
        return False
    return key.endswith(("_path", "_paths", "_dir"))


def _clear_structured_outputs(paths: RunPaths, *, keep_metadata: bool = False) -> None:
    for path in (
        paths.structured_prediction_path,
        paths.structured_raw_path,
    ):
        if path.exists():
            path.unlink()
    if not keep_metadata and paths.structured_metadata_path.exists():
        paths.structured_metadata_path.unlink()


def _validate_structured_prediction(
    prediction: dict[str, Any], *, source_text: str | None = None
) -> dict[str, Any]:
    reasons: list[str] = []
    scalar_field_names = (
        "document_type",
        "title",
        "institution",
        "date",
        "language",
        "summary_line",
    )
    normalized_scalars = [
        collapse_whitespace(str(prediction.get(name, ""))).lower() for name in scalar_field_names
    ]
    repeated_placeholder = {value for value in normalized_scalars if value}
    if len(repeated_placeholder) == 1:
        placeholder = next(iter(repeated_placeholder))
        if placeholder in {"document", "unknown", "n/a", "none", "null"}:
            reasons.append(f"all scalar fields collapsed to placeholder value {placeholder!r}")

    authors = prediction.get("authors") or []
    if not isinstance(authors, list):
        authors = [str(authors)]
    suspicious_author_tokens = {"[", "]", "{", "}", "[{", "}]", "```", "json"}
    if any(collapse_whitespace(str(author)) in suspicious_author_tokens for author in authors):
        reasons.append("authors field contains JSON fence or bracket fragments")

    summary_line = collapse_whitespace(str(prediction.get("summary_line", "")))
    if summary_line.lower().startswith("required:"):
        reasons.append("summary_line appears to echo extraction instructions")

    if source_text:
        normalized_source_text = _normalize_validation_text(source_text)
        for author in authors:
            author_text = collapse_whitespace(str(author))
            if author_text and not _text_present_in_source(author_text, normalized_source_text):
                reasons.append(f"author {author_text!r} not found in OCR text")

        for field_name in ("institution", "date"):
            field_value = collapse_whitespace(str(prediction.get(field_name, "")))
            if field_value and not _text_present_in_source(field_value, normalized_source_text):
                reasons.append(f"{field_name} value {field_value!r} not found in OCR text")

    return {"ok": not reasons, "reasons": reasons}


def _normalize_validation_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _text_present_in_source(value: str, normalized_source_text: str) -> bool:
    normalized_value = _normalize_validation_text(value)
    if not normalized_value:
        return True
    return normalized_value in normalized_source_text
