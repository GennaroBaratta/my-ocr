from __future__ import annotations

from pathlib import Path
import shutil

from my_ocr.application.ports import JsonWriter, OcrEngine

from ._run_state import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LAYOUT_DEVICE,
    DEFAULT_RUN_ROOT,
    RunPaths,
    clear_prediction_outputs,
    create_staged_ocr_paths,
    load_existing_input_path,
    publish_staged_ocr_run,
    run_ocr,
    write_json,
    write_run_metadata,
)
from my_ocr.application.artifacts.page_rerun_merge import (
    copy_reviewed_ocr_artifact_snapshot,
    merge_ocr_page_artifacts,
    resolve_page_path_for_number,
)


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
    page_path = resolve_page_path_for_number(paths, page_number)
    recorded_input_path = load_existing_input_path(paths)

    staged_paths = create_staged_ocr_paths(paths)
    partial_paths = create_staged_ocr_paths(paths)
    try:
        copy_reviewed_ocr_artifact_snapshot(paths, staged_paths)
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
        merge_ocr_page_artifacts(staged_paths, partial_paths, paths, page_number)
        publish_staged_ocr_run(
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
    clear_prediction_outputs(paths)
    return paths.run_dir


__all__ = ["run_reviewed_ocr_page_workflow"]
