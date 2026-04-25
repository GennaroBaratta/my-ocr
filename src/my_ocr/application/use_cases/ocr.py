from __future__ import annotations

from importlib import import_module
import shutil
from pathlib import Path
from typing import Any, Callable

from my_ocr.application.ports import (
    DocumentNormalizer,
    JsonWriter,
    MarkdownExtractor,
    OcrEngine,
    StructuredExtractor,
)
from my_ocr.application.services.rules_extractor import extract_from_markdown
from my_ocr.application.services.structured_validation import validate_structured_prediction
from my_ocr.application.use_cases.evaluation import evaluate_directories, write_markdown_report
from ._workflow_page_merge import (
    copy_review_artifact_snapshot as _copy_review_artifact_snapshot,
    copy_reviewed_ocr_artifact_snapshot as _copy_reviewed_ocr_artifact_snapshot,
    merge_ocr_page_artifacts as _merge_ocr_page_artifacts,
    merge_review_page_artifacts as _merge_review_page_artifacts,
    resolve_page_path_for_number as _resolve_page_path_for_number,
)
from ._workflow_payloads import (
    normalize_published_ocr_artifacts as _normalize_published_ocr_artifacts_impl,
)
from ._workflow_publish import (
    copy_path as _copy_path_impl,
    create_staged_ocr_paths as _create_staged_ocr_paths_impl,
    create_staged_review_paths as _create_staged_review_paths_impl,
    move_path as _move_path_impl,
    publish_staged_ocr_run as _publish_staged_ocr_run_impl,
    publish_staged_review_run as _publish_staged_review_run_impl,
    remove_paths as _remove_paths_impl,
    remove_replaced_ocr_outputs as _remove_replaced_ocr_outputs_impl,
)

DEFAULT_RUN_ROOT = "data/runs"
DEFAULT_CONFIG_PATH = "config/local.yaml"
DEFAULT_LAYOUT_DEVICE = "cuda"


class RunPaths:
    @classmethod
    def from_input(cls, *args: Any, **kwargs: Any) -> Any:
        return _run_paths_cls().from_input(*args, **kwargs)

    @classmethod
    def from_named_run(cls, *args: Any, **kwargs: Any) -> Any:
        return _run_paths_cls().from_named_run(*args, **kwargs)

    @classmethod
    def from_run_dir(cls, *args: Any, **kwargs: Any) -> Any:
        return _run_paths_cls().from_run_dir(*args, **kwargs)


def _run_paths_cls() -> Any:
    return import_module("my_ocr.adapters.outbound.filesystem.run_paths").RunPaths


def normalize_document(input_path: str, run_dir: str | Path) -> list[str]:
    adapter = import_module("my_ocr.adapters.outbound.filesystem.ingestion")
    return adapter.normalize_document(input_path, run_dir)


def run_ocr(*args: Any, **kwargs: Any) -> dict[str, Any]:
    adapter = import_module("my_ocr.adapters.outbound.ocr.glmocr_engine")
    return adapter.run_ocr(*args, **kwargs)


def prepare_review_artifacts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    adapter = import_module("my_ocr.adapters.outbound.ocr.glmocr_engine")
    return adapter.prepare_review_artifacts(*args, **kwargs)


def load_json(path: str | Path) -> Any:
    adapter = import_module("my_ocr.adapters.outbound.filesystem.json_store")
    return adapter.load_json(path)


def write_json(path: str | Path, payload: Any) -> None:
    adapter = import_module("my_ocr.adapters.outbound.filesystem.json_store")
    adapter.write_json(path, payload)


def run_ocr_workflow(
    input_path: str,
    *,
    run: str | None = None,
    run_root: str = DEFAULT_RUN_ROOT,
    config_path: str = DEFAULT_CONFIG_PATH,
    layout_device: str = DEFAULT_LAYOUT_DEVICE,
    layout_profile: str | None = "auto",
    normalize_document_fn: DocumentNormalizer = normalize_document,
    run_ocr_fn: OcrEngine = run_ocr,
    write_json_fn: JsonWriter = write_json,
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
    normalize_document_fn: DocumentNormalizer = normalize_document,
    prepare_review_artifacts_fn: OcrEngine = prepare_review_artifacts,
    write_json_fn: JsonWriter = write_json,
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
    run_ocr_fn: OcrEngine = run_ocr,
    write_json_fn: JsonWriter = write_json,
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
    prepare_review_artifacts_fn: OcrEngine = prepare_review_artifacts,
    write_json_fn: JsonWriter = write_json,
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
            page_numbers=[page_number],
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
    run_ocr_fn: OcrEngine = run_ocr,
    write_json_fn: JsonWriter = write_json,
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
            page_numbers=[page_number],
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
    extract_from_markdown_fn: MarkdownExtractor = extract_from_markdown,
    write_json_fn: JsonWriter = write_json,
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
        _clear_structured_outputs(paths, keep_metadata=True)
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
    write_json_fn: JsonWriter = write_json,
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


def _create_staged_ocr_paths(paths: Any) -> Any:
    return _create_staged_ocr_paths_impl(paths, RunPaths)


def _create_staged_review_paths(paths: Any) -> Any:
    return _create_staged_review_paths_impl(paths, RunPaths)


def _publish_staged_ocr_run(
    source_paths: Any,
    target_paths: Any,
    *,
    include_pages: bool = True,
    post_publish: Callable[[], None] | None = None,
) -> None:
    _publish_staged_ocr_run_impl(
        source_paths,
        target_paths,
        run_paths_cls=RunPaths,
        include_pages=include_pages,
        post_publish=post_publish,
        move_path_fn=_move_path,
        copy_path_fn=_copy_path,
        remove_paths_fn=_remove_paths,
        normalize_ocr_fn=_normalize_published_ocr_artifacts,
    )


def _publish_staged_review_run(
    source_paths: Any,
    target_paths: Any,
    *,
    post_publish: Callable[[], None] | None = None,
) -> None:
    _publish_staged_review_run_impl(
        source_paths,
        target_paths,
        run_paths_cls=RunPaths,
        post_publish=post_publish,
        move_path_fn=_move_path,
        copy_path_fn=_copy_path,
        remove_paths_fn=_remove_paths,
    )


def _move_path(source: Path, target: Path) -> None:
    _move_path_impl(source, target)


def _copy_path(source: Path, target: Path) -> None:
    _copy_path_impl(source, target)


def _remove_paths(paths: list[Path]) -> None:
    _remove_paths_impl(paths)


def _remove_replaced_ocr_outputs(paths: Any) -> None:
    _remove_replaced_ocr_outputs_impl(paths)


def _normalize_published_ocr_artifacts(
    paths: Any, *, source_run_dir: str | Path, target_run_dir: str | Path
) -> None:
    _normalize_published_ocr_artifacts_impl(
        paths,
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
    )


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
