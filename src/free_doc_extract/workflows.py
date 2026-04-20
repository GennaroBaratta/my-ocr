from __future__ import annotations

from importlib import import_module
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .evaluate import evaluate_directories, write_markdown_report
from .extract_rules import extract_from_markdown
from .ingest import normalize_document
from .ocr_fallback import detect_bbox_coord_space, extract_layout_blocks
from .ocr import prepare_review_artifacts, run_ocr
from .paths import RunPaths
from .review_artifacts import (
    REVIEW_LAYOUT_VERSION,
    build_review_page_from_layout,
    build_review_layout_payload,
    load_review_layout_payload,
    review_layout_pages_by_number,
    save_review_layout_payload,
)
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
    layout_profile: str | None = "auto",
    normalize_document_fn: Callable[[str, str | Path], list[str]] = normalize_document,
    run_ocr_fn: Callable[..., dict[str, Any]] = run_ocr,
    write_json_fn: Callable[[str | Path, Any], None] = write_json,
    recorded_input_path: str | None = None,
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
            layout_profile=layout_profile,
        )
        _publish_staged_ocr_run(
            staged_paths,
            paths,
            post_publish=lambda: write_run_metadata(
                paths.run_dir,
                recorded_input_path or input_path,
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


def prepare_review_workflow(
    input_path: str,
    *,
    run: str | None = None,
    run_root: str = DEFAULT_RUN_ROOT,
    config_path: str = DEFAULT_CONFIG_PATH,
    layout_device: str = DEFAULT_LAYOUT_DEVICE,
    layout_profile: str | None = "auto",
    normalize_document_fn: Callable[[str, str | Path], list[str]] = normalize_document,
    prepare_review_artifacts_fn: Callable[..., dict[str, Any]] = prepare_review_artifacts,
    write_json_fn: Callable[[str | Path, Any], None] = write_json,
) -> Path:
    paths = RunPaths.from_input(input_path, run=run, run_root=run_root)
    run_dir_preexisted = paths.run_dir.exists()
    paths.ensure_run_dir()
    try:
        staged_paths = _create_staged_review_paths(paths)
        page_paths = normalize_document_fn(input_path, staged_paths.run_dir)
        result = prepare_review_artifacts_fn(
            page_paths,
            staged_paths.run_dir,
            config_path=config_path,
            layout_device=layout_device,
            layout_profile=layout_profile,
        )
        _publish_staged_review_run(
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
    _clear_prediction_outputs(paths)
    return paths.run_dir


def run_reviewed_ocr_workflow(
    run: str,
    *,
    run_root: str = DEFAULT_RUN_ROOT,
    config_path: str = DEFAULT_CONFIG_PATH,
    layout_device: str = DEFAULT_LAYOUT_DEVICE,
    layout_profile: str | None = "auto",
    run_ocr_fn: Callable[..., dict[str, Any]] = run_ocr,
    write_json_fn: Callable[[str | Path, Any], None] = write_json,
) -> Path:
    paths = RunPaths.from_named_run(run, run_root=run_root)
    page_paths = paths.list_page_paths()
    if not page_paths:
        raise FileNotFoundError(f"No page images found in {paths.pages_dir}")
    recorded_input_path = _load_existing_input_path(paths)

    staged_paths = _create_staged_ocr_paths(paths)
    try:
        result = run_ocr_fn(
            page_paths,
            staged_paths.run_dir,
            config_path=config_path,
            layout_device=layout_device,
            layout_profile=layout_profile,
            reviewed_layout_path=(
                paths.reviewed_layout_path if paths.reviewed_layout_path.exists() else None
            ),
        )
        _publish_staged_ocr_run(
            staged_paths,
            paths,
            include_pages=False,
            post_publish=lambda: write_run_metadata(
                paths.run_dir,
                recorded_input_path,
                paths.list_page_paths(),
                {
                    **result,
                    "raw_dir": str(paths.raw_dir),
                },
                write_json_fn=write_json_fn,
            ),
        )
    finally:
        shutil.rmtree(staged_paths.run_dir, ignore_errors=True)
    _clear_prediction_outputs(paths)
    return paths.run_dir


def prepare_review_page_workflow(
    run: str,
    page_number: int,
    *,
    run_root: str = DEFAULT_RUN_ROOT,
    config_path: str = DEFAULT_CONFIG_PATH,
    layout_device: str = DEFAULT_LAYOUT_DEVICE,
    layout_profile: str | None = "auto",
    prepare_review_artifacts_fn: Callable[..., dict[str, Any]] = prepare_review_artifacts,
    write_json_fn: Callable[[str | Path, Any], None] = write_json,
) -> Path:
    paths = RunPaths.from_named_run(run, run_root=run_root)
    page_path = _resolve_page_path_for_number(paths, page_number)
    recorded_input_path = _load_existing_input_path(paths)

    staged_paths = _create_staged_review_paths(paths)
    partial_paths = _create_staged_review_paths(paths)
    try:
        _copy_review_artifact_snapshot(paths, staged_paths)
        result = prepare_review_artifacts_fn(
            [page_path],
            partial_paths.run_dir,
            config_path=config_path,
            layout_device=layout_device,
            layout_profile=layout_profile,
        )
        _merge_review_page_artifacts(paths, staged_paths, partial_paths, page_number)
        _publish_staged_review_run(
            staged_paths,
            paths,
            post_publish=lambda: write_run_metadata(
                paths.run_dir,
                recorded_input_path,
                paths.list_page_paths(),
                {
                    **result,
                    "raw_dir": str(paths.raw_dir),
                },
                write_json_fn=write_json_fn,
            ),
        )
    finally:
        shutil.rmtree(partial_paths.run_dir, ignore_errors=True)
        shutil.rmtree(staged_paths.run_dir, ignore_errors=True)
    _clear_prediction_outputs(paths)
    return paths.run_dir


def run_reviewed_ocr_page_workflow(
    run: str,
    page_number: int,
    *,
    run_root: str = DEFAULT_RUN_ROOT,
    config_path: str = DEFAULT_CONFIG_PATH,
    layout_device: str = DEFAULT_LAYOUT_DEVICE,
    layout_profile: str | None = "auto",
    run_ocr_fn: Callable[..., dict[str, Any]] = run_ocr,
    write_json_fn: Callable[[str | Path, Any], None] = write_json,
) -> Path:
    paths = RunPaths.from_named_run(run, run_root=run_root)
    page_path = _resolve_page_path_for_number(paths, page_number)
    recorded_input_path = _load_existing_input_path(paths)

    staged_paths = _create_staged_ocr_paths(paths)
    partial_paths = _create_staged_ocr_paths(paths)
    try:
        _copy_reviewed_ocr_artifact_snapshot(paths, staged_paths)
        result = run_ocr_fn(
            [page_path],
            partial_paths.run_dir,
            config_path=config_path,
            layout_device=layout_device,
            layout_profile=layout_profile,
            reviewed_layout_path=(
                paths.reviewed_layout_path if paths.reviewed_layout_path.exists() else None
            ),
        )
        _merge_ocr_page_artifacts(staged_paths, partial_paths, paths, page_number)
        _publish_staged_ocr_run(
            staged_paths,
            paths,
            include_pages=False,
            post_publish=lambda: write_run_metadata(
                paths.run_dir,
                recorded_input_path,
                paths.list_page_paths(),
                {
                    **result,
                    "raw_dir": str(paths.raw_dir),
                },
                write_json_fn=write_json_fn,
            ),
        )
    finally:
        shutil.rmtree(partial_paths.run_dir, ignore_errors=True)
        shutil.rmtree(staged_paths.run_dir, ignore_errors=True)
    _clear_prediction_outputs(paths)
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
    layout_profile: str | None = "auto",
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
    layout_diagnostics = ocr_result.get("layout_diagnostics")
    if layout_diagnostics:
        payload["layout_diagnostics"] = layout_diagnostics
    if paths.reviewed_layout_path.exists():
        payload["reviewed_layout_path"] = str(paths.reviewed_layout_path)
    write_json_fn(paths.meta_path, payload)


def _create_staged_ocr_paths(paths: RunPaths) -> RunPaths:
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{paths.run_name}-ocr-", dir=str(paths.run_dir.parent))
    )
    return RunPaths.from_run_dir(staging_dir)


def _create_staged_review_paths(paths: RunPaths) -> RunPaths:
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{paths.run_name}-review-", dir=str(paths.run_dir.parent))
    )
    return RunPaths.from_run_dir(staging_dir)


def _publish_staged_ocr_run(
    source_paths: RunPaths,
    target_paths: RunPaths,
    *,
    include_pages: bool = True,
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
        _move_ocr_artifacts(
            target_paths,
            backup_paths,
            moved_pairs=backed_up_pairs,
            include_pages=include_pages,
        )
        _move_ocr_artifacts(
            source_paths,
            target_paths,
            moved_pairs=published_pairs,
            include_pages=include_pages,
        )
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


def _publish_staged_review_run(
    source_paths: RunPaths,
    target_paths: RunPaths,
    *,
    post_publish: Callable[[], None] | None = None,
) -> None:
    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_paths.run_name}-review-backup-", dir=str(target_paths.run_dir.parent)
        )
    )
    backup_paths = RunPaths.from_run_dir(backup_dir)
    backed_up_pairs: list[tuple[Path, Path]] = []
    published_pairs: list[tuple[Path, Path]] = []
    try:
        if target_paths.meta_path.exists():
            _move_path(target_paths.meta_path, backup_paths.meta_path)
            backed_up_pairs.append((target_paths.meta_path, backup_paths.meta_path))
        _move_review_artifacts(target_paths, backup_paths, moved_pairs=backed_up_pairs)
        _move_review_artifacts(source_paths, target_paths, moved_pairs=published_pairs)
        _normalize_published_review_json(
            target_paths.reviewed_layout_path,
            source_run_dir=source_paths.run_dir,
            target_run_dir=target_paths.run_dir,
        )
        if post_publish is not None:
            post_publish()
        _remove_replaced_ocr_outputs(target_paths)
    except Exception:
        try:
            _remove_paths([target_paths.meta_path, *[target for _, target in published_pairs]])
            _restore_moved_artifacts(backed_up_pairs)
        except Exception as restore_exc:
            raise RuntimeError(
                "Failed to restore review artifacts after publish failure. "
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
    include_pages: bool = True,
) -> None:
    for source, target in _ocr_artifact_pairs(
        source_paths,
        target_paths,
        include_pages=include_pages,
    ):
        if not source.exists():
            continue
        _move_path(source, target)
        if moved_pairs is not None:
            moved_pairs.append((source, target))


def _move_review_artifacts(
    source_paths: RunPaths,
    target_paths: RunPaths,
    *,
    moved_pairs: list[tuple[Path, Path]] | None = None,
) -> None:
    for source, target in _review_artifact_pairs(source_paths, target_paths):
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


def _remove_replaced_ocr_outputs(paths: RunPaths) -> None:
    for target in (
        paths.fallback_dir,
        paths.ocr_markdown_path,
        paths.ocr_json_path,
        paths.ocr_fallback_path,
    ):
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


def _ocr_artifact_pairs(
    source_paths: RunPaths,
    target_paths: RunPaths,
    *,
    include_pages: bool = True,
) -> list[tuple[Path, Path]]:
    source_artifacts = (
        source_paths.published_ocr_artifact_paths()
        if include_pages
        else source_paths.published_reviewed_ocr_artifact_paths()
    )
    target_artifacts = (
        target_paths.published_ocr_artifact_paths()
        if include_pages
        else target_paths.published_reviewed_ocr_artifact_paths()
    )
    return list(
        zip(
            source_artifacts,
            target_artifacts,
            strict=True,
        )
    )


def _review_artifact_pairs(
    source_paths: RunPaths, target_paths: RunPaths
) -> list[tuple[Path, Path]]:
    return list(
        zip(
            source_paths.published_review_artifact_paths(),
            target_paths.published_review_artifact_paths(),
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


def _normalize_published_review_json(
    payload_path: Path, *, source_run_dir: str | Path, target_run_dir: str | Path
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
            _rewrite_path_keys(
                page, ("page_path", "source_sdk_json_path"), source_prefix, target_prefix
            )
    write_json(payload_path, payload)


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


def _resolve_page_path_for_number(paths: RunPaths, page_number: int) -> str:
    for page_path in paths.list_page_paths():
        if _page_number_for_path(page_path) == page_number:
            return page_path
    raise FileNotFoundError(f"Page {page_number} not found in {paths.pages_dir}")


def _page_number_for_path(page_path: str | Path) -> int:
    match = re.fullmatch(r"page-(\d+)", Path(page_path).stem)
    if match is None:
        raise ValueError(f"Cannot determine page number from {page_path}")
    return int(match.group(1))


def _copy_review_artifact_snapshot(source_paths: RunPaths, staged_paths: RunPaths) -> None:
    staged_paths.ensure_run_dir()
    for source, target in (
        (source_paths.pages_dir, staged_paths.pages_dir),
        (source_paths.raw_dir, staged_paths.raw_dir),
        (source_paths.reviewed_layout_path, staged_paths.reviewed_layout_path),
    ):
        if source.exists():
            _copy_path(source, target)


def _copy_reviewed_ocr_artifact_snapshot(source_paths: RunPaths, staged_paths: RunPaths) -> None:
    staged_paths.ensure_run_dir()
    for source, target in (
        (source_paths.raw_dir, staged_paths.raw_dir),
        (source_paths.fallback_dir, staged_paths.fallback_dir),
        (source_paths.ocr_markdown_path, staged_paths.ocr_markdown_path),
        (source_paths.ocr_json_path, staged_paths.ocr_json_path),
        (source_paths.ocr_fallback_path, staged_paths.ocr_fallback_path),
    ):
        if source.exists():
            _copy_path(source, target)


def _merge_review_page_artifacts(
    source_paths: RunPaths,
    staged_paths: RunPaths,
    partial_paths: RunPaths,
    page_number: int,
) -> None:
    partial_payload = load_review_layout_payload(partial_paths.reviewed_layout_path)
    if partial_payload is None:
        raise FileNotFoundError(f"Missing partial reviewed layout: {partial_paths.reviewed_layout_path}")

    partial_page = review_layout_pages_by_number(partial_payload).get(page_number)
    if partial_page is None:
        raise KeyError(f"Partial reviewed layout is missing page {page_number}")

    _rewrite_path_keys(
        partial_page,
        ("page_path", "source_sdk_json_path"),
        str(partial_paths.run_dir),
        str(staged_paths.run_dir),
    )
    _copy_page_dir(partial_paths.raw_page_dir(page_number), staged_paths.raw_page_dir(page_number))

    existing_payload = load_review_layout_payload(staged_paths.reviewed_layout_path) or {
        "version": REVIEW_LAYOUT_VERSION,
        "status": "prepared",
        "pages": [],
        "summary": {"page_count": 0},
    }
    existing_pages = existing_payload.get("pages")
    if not isinstance(existing_pages, list):
        existing_pages = []
    existing_by_number = _seed_review_pages_for_merge(
        source_paths,
        staged_paths,
        review_layout_pages_by_number(existing_payload),
    )
    existing_by_number[page_number] = partial_page
    ordered_pages = _ordered_page_records(existing_pages, existing_by_number, staged_paths.pages_dir)
    status = partial_payload.get("status")
    payload = build_review_layout_payload(
        ordered_pages,
        status=status if isinstance(status, str) and status.strip() else "prepared",
    )
    save_review_layout_payload(staged_paths.reviewed_layout_path, payload)


def _seed_review_pages_for_merge(
    source_paths: RunPaths,
    staged_paths: RunPaths,
    review_pages_by_number: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    seeded_pages = dict(review_pages_by_number)
    expected_page_numbers = [
        _page_number_for_path(page_path) for page_path in staged_paths.list_page_paths()
    ]
    missing_page_numbers = [
        page_number for page_number in expected_page_numbers if page_number not in seeded_pages
    ]
    if not missing_page_numbers:
        return seeded_pages

    ocr_payload = load_json(source_paths.ocr_json_path)
    if not isinstance(ocr_payload, dict):
        raise FileNotFoundError(
            "Missing OCR payload required to preserve untouched review pages: "
            f"{source_paths.ocr_json_path}"
        )

    ocr_pages_by_number = _pages_by_number(ocr_payload)
    for missing_page_number in missing_page_numbers:
        ocr_page = ocr_pages_by_number.get(missing_page_number)
        if ocr_page is None:
            raise KeyError(
                "OCR payload is missing untouched page "
                f"{missing_page_number} required for review merge"
            )
        seeded_pages[missing_page_number] = _build_review_page_from_ocr_page(
            source_paths,
            staged_paths,
            missing_page_number,
            ocr_page,
        )
    return seeded_pages


def _build_review_page_from_ocr_page(
    source_paths: RunPaths,
    staged_paths: RunPaths,
    page_number: int,
    ocr_page: dict[str, Any],
) -> dict[str, Any]:
    sdk_json_path = _resolve_sdk_json_path_for_page(source_paths, page_number, ocr_page)
    if not sdk_json_path.exists():
        raise FileNotFoundError(
            f"Missing SDK JSON for untouched page {page_number}: {sdk_json_path}"
        )

    page_path = _resolve_page_path_for_number(staged_paths, page_number)
    _copy_path(sdk_json_path.parent, staged_paths.raw_page_dir(page_number))
    staged_sdk_json_path = staged_paths.raw_page_dir(page_number) / sdk_json_path.name
    layout = load_json(sdk_json_path)
    blocks = extract_layout_blocks(layout)
    coord_space = _detect_coord_space(blocks, page_path)
    image_width, image_height = _get_image_size(page_path)
    return build_review_page_from_layout(
        page_number=page_number,
        page_path=page_path,
        source_sdk_json_path=str(staged_sdk_json_path),
        layout=layout,
        coord_space=coord_space,
        image_width=image_width,
        image_height=image_height,
    )


def _resolve_sdk_json_path_for_page(
    source_paths: RunPaths,
    page_number: int,
    ocr_page: dict[str, Any],
) -> Path:
    sdk_json_path = ocr_page.get("sdk_json_path")
    if not isinstance(sdk_json_path, str) or not sdk_json_path.strip():
        raise KeyError(f"OCR payload is missing sdk_json_path for untouched page {page_number}")

    candidate = Path(sdk_json_path)
    if candidate.exists():
        return candidate
    if candidate.is_absolute():
        raise FileNotFoundError(candidate)

    resolved = source_paths.run_dir / candidate
    if resolved.exists():
        return resolved
    raise FileNotFoundError(resolved)


def _get_image_size(page_path: str | Path) -> tuple[int, int]:
    try:
        image_module = import_module("PIL.Image")
    except ImportError as exc:
        raise RuntimeError("Review page merge requires Pillow.") from exc

    with image_module.open(page_path) as image:
        return image.size


def _detect_coord_space(blocks: list[dict[str, Any]], page_path: str | Path) -> str:
    if not blocks:
        return "unknown"

    try:
        image_width, image_height = _get_image_size(page_path)
    except OSError:
        return detect_bbox_coord_space(blocks)

    return detect_bbox_coord_space(blocks, width=image_width, height=image_height)


def _merge_ocr_page_artifacts(
    staged_paths: RunPaths,
    partial_paths: RunPaths,
    target_paths: RunPaths,
    page_number: int,
) -> None:
    partial_payload = load_json(partial_paths.ocr_json_path)
    if not isinstance(partial_payload, dict):
        raise FileNotFoundError(f"Missing partial OCR payload: {partial_paths.ocr_json_path}")

    partial_pages = _pages_by_number(partial_payload)
    partial_page = partial_pages.get(page_number)
    if partial_page is None:
        raise KeyError(f"Partial OCR payload is missing page {page_number}")

    _rewrite_path_keys(
        partial_page,
        ("page_path", "sdk_json_path"),
        str(partial_paths.run_dir),
        str(staged_paths.run_dir),
    )
    _copy_page_dir(partial_paths.raw_page_dir(page_number), staged_paths.raw_page_dir(page_number))

    existing_payload = load_json(staged_paths.ocr_json_path)
    if not isinstance(existing_payload, dict):
        raise FileNotFoundError(f"Missing existing OCR payload: {staged_paths.ocr_json_path}")
    existing_pages = existing_payload.get("pages")
    if not isinstance(existing_pages, list):
        raise ValueError(f"Invalid OCR pages payload: {staged_paths.ocr_json_path}")

    existing_by_number = _pages_by_number(existing_payload)
    existing_by_number[page_number] = partial_page
    merged_pages = _ordered_page_records(existing_pages, existing_by_number, target_paths.pages_dir)

    merged_payload = dict(existing_payload)
    merged_payload["pages"] = merged_pages
    merged_summary = {
        "page_count": len(merged_pages),
        "sources": _count_page_sources(merged_pages),
    }
    if target_paths.reviewed_layout_path.exists():
        merged_summary["reviewed_layout"] = {
            "path": str(target_paths.reviewed_layout_path),
            "page_count": len(merged_pages),
            "apply_mode": "planning_and_fallback_only",
        }
    merged_payload["summary"] = merged_summary
    write_json(staged_paths.ocr_json_path, merged_payload)
    _write_merged_markdown(staged_paths.ocr_markdown_path, merged_pages)
    _merge_fallback_page_artifacts(staged_paths, partial_paths, existing_pages, page_number)


def _merge_fallback_page_artifacts(
    staged_paths: RunPaths,
    partial_paths: RunPaths,
    existing_pages: list[Any],
    page_number: int,
) -> None:
    partial_payload = load_json(partial_paths.ocr_fallback_path) if partial_paths.ocr_fallback_path.exists() else {}
    partial_pages = _pages_by_number(partial_payload if isinstance(partial_payload, dict) else {})
    partial_page = partial_pages.get(page_number)
    partial_fallback_dir = partial_paths.fallback_dir / f"page-{page_number:04d}"
    staged_fallback_dir = staged_paths.fallback_dir / f"page-{page_number:04d}"
    _remove_paths([staged_fallback_dir])
    if partial_fallback_dir.exists():
        _copy_page_dir(partial_fallback_dir, staged_fallback_dir)

    existing_payload = load_json(staged_paths.ocr_fallback_path) if staged_paths.ocr_fallback_path.exists() else {}
    existing_by_number = _pages_by_number(existing_payload if isinstance(existing_payload, dict) else {})
    if partial_page is None:
        existing_by_number.pop(page_number, None)
    else:
        _rewrite_path_keys(
            partial_page,
            ("page_path",),
            str(partial_paths.run_dir),
            str(staged_paths.run_dir),
        )
        chunks = partial_page.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, dict):
                    _rewrite_path_keys(
                        chunk,
                        ("crop_path", "text_path"),
                        str(partial_paths.run_dir),
                        str(staged_paths.run_dir),
                    )
        existing_by_number[page_number] = partial_page

    merged_pages = _ordered_page_records(existing_pages, existing_by_number, staged_paths.pages_dir)
    if not merged_pages:
        _remove_paths([staged_paths.ocr_fallback_path, staged_paths.fallback_dir])
        return
    write_json(
        staged_paths.ocr_fallback_path,
        {"pages": merged_pages, "summary": {"page_count": len(merged_pages)}},
    )


def _ordered_page_records(
    existing_pages: list[Any],
    pages_by_number: dict[int, dict[str, Any]],
    pages_dir: Path,
) -> list[dict[str, Any]]:
    ordered_numbers: list[int] = []
    for page in existing_pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        if isinstance(page_number, int) and page_number > 0 and page_number not in ordered_numbers:
            ordered_numbers.append(page_number)
    for page_path in sorted(str(path) for path in pages_dir.iterdir() if path.is_file()) if pages_dir.exists() else []:
        page_number = _page_number_for_path(page_path)
        if page_number not in ordered_numbers:
            ordered_numbers.append(page_number)
    for page_number in sorted(pages_by_number):
        if page_number not in ordered_numbers:
            ordered_numbers.append(page_number)
    return [pages_by_number[page_number] for page_number in ordered_numbers if page_number in pages_by_number]


def _pages_by_number(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return {}
    indexed: dict[int, dict[str, Any]] = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        if isinstance(page_number, int) and page_number > 0:
            indexed[page_number] = page
    return indexed


def _count_page_sources(pages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for page in pages:
        source = page.get("markdown_source")
        if not isinstance(source, str) or not source:
            continue
        counts[source] = counts.get(source, 0) + 1
    return counts


def _write_merged_markdown(path: Path, pages: list[dict[str, Any]]) -> None:
    markdown_parts = []
    for page in pages:
        markdown = page.get("markdown")
        if isinstance(markdown, str) and markdown.strip():
            markdown_parts.append(markdown.strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(markdown_parts), encoding="utf-8")


def _copy_page_dir(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing staged page directory: {source}")
    _copy_path(source, target)


def _clear_structured_outputs(paths: RunPaths, *, keep_metadata: bool = False) -> None:
    for path in (
        paths.structured_prediction_path,
        paths.structured_raw_path,
    ):
        if path.exists():
            path.unlink()
    if not keep_metadata and paths.structured_metadata_path.exists():
        paths.structured_metadata_path.unlink()


def _clear_prediction_outputs(paths: RunPaths) -> None:
    for path in (
        paths.rules_prediction_path,
        paths.canonical_prediction_path,
    ):
        if path.exists():
            path.unlink()
    _clear_structured_outputs(paths)


def _remove_dir_if_empty(path: str | Path) -> None:
    target = Path(path)
    if target.exists() and not any(target.iterdir()):
        target.rmdir()


def _load_existing_input_path(paths: RunPaths) -> str:
    if not paths.meta_path.exists():
        return str(paths.run_dir)
    payload = load_json(paths.meta_path)
    if not isinstance(payload, dict):
        return str(paths.run_dir)
    input_path = payload.get("input_path")
    return str(input_path) if input_path else str(paths.run_dir)


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
