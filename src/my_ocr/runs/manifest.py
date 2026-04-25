from __future__ import annotations

from pathlib import Path

from my_ocr.support.filesystem import read_json, write_json
from my_ocr.domain import (
    RunManifest,
    RunNotFound,
)
from my_ocr.runs.artifact_io import RunLayoutPaths


def write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    write_json(RunLayoutPaths(run_dir).manifest, manifest.model_dump(mode="json"))


def load_manifest(run_dir: Path) -> RunManifest:
    if not run_dir.exists():
        raise RunNotFound(f"Run not found: {run_dir.name}")
    manifest_path = RunLayoutPaths(run_dir).manifest
    if not manifest_path.exists():
        raise RunNotFound(f"Run manifest not found: {run_dir.name}")
    payload = read_json(manifest_path)
    manifest = RunManifest.model_validate(payload)
    return manifest.model_copy(
        update={
            "pages": [
                page.model_copy(update={"resolved_path": run_dir / page.image_path})
                for page in manifest.pages
            ]
        }
    )
