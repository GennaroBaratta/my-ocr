from __future__ import annotations

from pathlib import Path
import shutil

from my_ocr.application.ports import DocumentNormalizer, JsonWriter, OcrEngine

from ._run_state import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LAYOUT_DEVICE,
    DEFAULT_RUN_ROOT,
    RunPaths,
    create_staged_ocr_paths,
    normalize_document,
    publish_staged_ocr_run,
    remove_dir_if_empty,
    run_ocr,
    write_json,
    write_run_metadata,
)


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
        staged_paths = create_staged_ocr_paths(paths)
        page_paths = normalize_document_fn(input_path, staged_paths.run_dir)
        result = run_ocr_fn(
            page_paths,
            staged_paths.run_dir,
            config_path=config_path,
            layout_device=layout_device,
            layout_profile=layout_profile,
        )
        publish_staged_ocr_run(
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
            remove_dir_if_empty(paths.run_dir)
        raise
    finally:
        staged_paths_local = locals().get("staged_paths")
        if staged_paths_local is not None:
            shutil.rmtree(staged_paths_local.run_dir, ignore_errors=True)
    return paths.run_dir


__all__ = ["run_ocr_workflow", "write_run_metadata"]
