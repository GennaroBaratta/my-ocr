from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from my_ocr.application.artifacts.run_paths import RunPaths as ArtifactRunPaths
from my_ocr.application.ports import JsonWriter

from my_ocr.application.artifacts.artifact_payload_paths import (
    normalize_published_ocr_artifacts as _normalize_published_ocr_artifacts_impl,
)
from my_ocr.application.artifacts.run_artifact_publisher import (
    copy_path as _copy_path_impl,
    create_staged_ocr_paths as _create_staged_ocr_paths_impl,
    create_staged_review_paths as _create_staged_review_paths_impl,
    move_path as _move_path_impl,
    publish_staged_ocr_run as _publish_staged_ocr_run_impl,
    publish_staged_review_run as _publish_staged_review_run_impl,
    remove_paths as _remove_paths_impl,
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
    return ArtifactRunPaths


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


def create_staged_ocr_paths(paths: Any) -> Any:
    return _create_staged_ocr_paths_impl(paths, RunPaths)


def create_staged_review_paths(paths: Any) -> Any:
    return _create_staged_review_paths_impl(paths, RunPaths)


def publish_staged_ocr_run(
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
        move_path_fn=move_path,
        copy_path_fn=copy_path,
        remove_paths_fn=remove_paths,
        normalize_ocr_fn=normalize_published_ocr_artifacts,
    )


def publish_staged_review_run(
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
        move_path_fn=move_path,
        copy_path_fn=copy_path,
        remove_paths_fn=remove_paths,
    )


def move_path(source: Path, target: Path) -> None:
    _move_path_impl(source, target)


def copy_path(source: Path, target: Path) -> None:
    _copy_path_impl(source, target)


def remove_paths(paths: list[Path]) -> None:
    _remove_paths_impl(paths)


def normalize_published_ocr_artifacts(
    paths: Any, *, source_run_dir: str | Path, target_run_dir: str | Path
) -> None:
    _normalize_published_ocr_artifacts_impl(
        paths,
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
    )


def clear_structured_outputs(paths: RunPaths, *, keep_metadata: bool = False) -> None:
    for path in (
        paths.structured_prediction_path,
        paths.structured_raw_path,
    ):
        if path.exists():
            path.unlink()
    if not keep_metadata and paths.structured_metadata_path.exists():
        paths.structured_metadata_path.unlink()


def clear_prediction_outputs(paths: RunPaths) -> None:
    for path in (
        paths.rules_prediction_path,
        paths.canonical_prediction_path,
    ):
        if path.exists():
            path.unlink()
    clear_structured_outputs(paths)


def remove_dir_if_empty(path: str | Path) -> None:
    target = Path(path)
    if target.exists() and not any(target.iterdir()):
        target.rmdir()


def load_existing_input_path(paths: RunPaths) -> str:
    if not paths.meta_path.exists():
        return str(paths.run_dir)
    payload = load_json(paths.meta_path)
    if not isinstance(payload, dict):
        return str(paths.run_dir)
    input_path = payload.get("input_path")
    return str(input_path) if input_path else str(paths.run_dir)
