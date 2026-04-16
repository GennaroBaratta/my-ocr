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
    run_dir_preexisted = paths.run_dir.exists()
    paths.ensure_run_dir()
    try:
        staged_paths = _create_staged_ocr_paths(paths)
        page_paths = normalize_document_fn(input_path, staged_paths.run_dir)
        result = run_ocr_fn(
            page_paths,
            staged_paths.run_dir,
            config_path=config_path,
            layout_device=layout_device,
        )
        _publish_staged_ocr_run(
            staged_paths,
            paths,
            post_publish=lambda: write_run_metadata(
                paths.run_dir,
                input_path,
                paths.list_page_paths(),
                {
                    **result,
                    "raw_dir": str(paths.raw_dir),
                },
                write_json_fn=write_json_fn,
            ),
        )
    except Exception:
        if not run_dir_preexisted:
            _remove_dir_if_empty(paths.run_dir)
        raise
    finally:
        staged_paths_local = locals().get("staged_paths")
        if staged_paths_local is not None:
            shutil.rmtree(staged_paths_local.run_dir, ignore_errors=True)
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


def _publish_staged_ocr_run(
    source_paths: RunPaths,
    target_paths: RunPaths,
    *,
    post_publish: Callable[[], None] | None = None,
) -> None:
    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_paths.run_name}-ocr-backup-", dir=str(target_paths.run_dir.parent)
        )
    )
    backup_paths = RunPaths.from_run_dir(backup_dir)
    backed_up_pairs: list[tuple[Path, Path]] = []
    published_pairs: list[tuple[Path, Path]] = []
    try:
        if target_paths.meta_path.exists():
            _move_path(target_paths.meta_path, backup_paths.meta_path)
            backed_up_pairs.append((target_paths.meta_path, backup_paths.meta_path))
        _move_ocr_artifacts(target_paths, backup_paths, moved_pairs=backed_up_pairs)
        _move_ocr_artifacts(source_paths, target_paths, moved_pairs=published_pairs)
        _normalize_published_ocr_artifacts(
            target_paths,
            source_run_dir=source_paths.run_dir,
            target_run_dir=target_paths.run_dir,
        )
        if post_publish is not None:
            post_publish()
    except Exception:
        try:
            _remove_paths([target_paths.meta_path, *[target for _, target in published_pairs]])
            _restore_moved_artifacts(backed_up_pairs)
        except Exception as restore_exc:
            raise RuntimeError(
                "Failed to restore OCR artifacts after publish failure. "
                f"Backup preserved at {backup_dir}."
            ) from restore_exc
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def _move_ocr_artifacts(
    source_paths: RunPaths,
    target_paths: RunPaths,
    *,
    moved_pairs: list[tuple[Path, Path]] | None = None,
) -> None:
    for source, target in _ocr_artifact_pairs(source_paths, target_paths):
        if not source.exists():
            continue
        _move_path(source, target)
        if moved_pairs is not None:
            moved_pairs.append((source, target))


def _restore_moved_artifacts(moved_pairs: list[tuple[Path, Path]]) -> None:
    for source, target in reversed(moved_pairs):
        _copy_path(target, source)


def _remove_ocr_artifacts(paths: RunPaths) -> None:
    for target in paths.published_ocr_artifact_paths():
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()


def _remove_paths(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def _ocr_artifact_pairs(source_paths: RunPaths, target_paths: RunPaths) -> list[tuple[Path, Path]]:
    return list(
        zip(
            source_paths.published_ocr_artifact_paths(),
            target_paths.published_ocr_artifact_paths(),
            strict=True,
        )
    )


def _move_path(source: Path, target: Path) -> None:
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)


def _copy_path(source: Path, target: Path) -> None:
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def _normalize_published_ocr_artifacts(
    paths: RunPaths, *, source_run_dir: str | Path, target_run_dir: str | Path
) -> None:
    _normalize_published_ocr_json(
        paths.ocr_json_path,
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
    )
    _normalize_published_fallback_json(
        paths.ocr_fallback_path,
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
    )


def _normalize_published_ocr_json(
    payload_path: Path, *, source_run_dir: str | Path, target_run_dir: str | Path
) -> None:
    _normalize_published_json_paths(
        payload_path,
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
        page_keys=("page_path", "sdk_json_path"),
    )


def _normalize_published_fallback_json(
    payload_path: Path, *, source_run_dir: str | Path, target_run_dir: str | Path
) -> None:
    _normalize_published_json_paths(
        payload_path,
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
        page_keys=("page_path",),
        chunk_keys=("crop_path", "text_path"),
    )


def _normalize_published_json_paths(
    payload_path: Path,
    *,
    source_run_dir: str | Path,
    target_run_dir: str | Path,
    page_keys: tuple[str, ...],
    chunk_keys: tuple[str, ...] = (),
) -> None:
    if not payload_path.exists():
        return

    payload = load_json(payload_path)
    if not isinstance(payload, dict):
        return

    source_prefix = str(source_run_dir)
    target_prefix = str(target_run_dir)
    pages = payload.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            _rewrite_path_keys(page, page_keys, source_prefix, target_prefix)
            chunks = page.get("chunks")
            if not chunk_keys or not isinstance(chunks, list):
                continue
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                _rewrite_path_keys(chunk, chunk_keys, source_prefix, target_prefix)
    write_json(payload_path, payload)


def _rewrite_path_keys(
    payload: dict[str, Any], keys: tuple[str, ...], source_prefix: str, target_prefix: str
) -> None:
    for key in keys:
        if key in payload:
            payload[key] = _rewrite_path_value(payload.get(key), source_prefix, target_prefix)


def _rewrite_path_value(value: Any, source_prefix: str, target_prefix: str) -> Any:
    if isinstance(value, str) and source_prefix in value:
        return value.replace(source_prefix, target_prefix)
    return value


def _clear_structured_outputs(paths: RunPaths, *, keep_metadata: bool = False) -> None:
    for path in (
        paths.structured_prediction_path,
        paths.structured_raw_path,
    ):
        if path.exists():
            path.unlink()
    if not keep_metadata and paths.structured_metadata_path.exists():
        paths.structured_metadata_path.unlink()


def _remove_dir_if_empty(path: str | Path) -> None:
    target = Path(path)
    if target.exists() and not any(target.iterdir()):
        target.rmdir()


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
