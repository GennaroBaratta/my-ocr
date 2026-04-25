from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from my_ocr.filesystem import read_json, write_json
from my_ocr.models import (
    RunManifest,
    RunNotFound,
    SCHEMA_VERSION,
    UnsupportedRunSchema,
)
from my_ocr.run_layout import RunLayoutPaths


UNSUPPORTED_V3_MESSAGE = (
    "Unsupported run schema: this run was created before v3. "
    "Re-run the document to create a v3 run."
)


def write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    write_json(RunLayoutPaths(run_dir).manifest, manifest.model_dump(mode="json"))


def load_manifest(run_dir: Path) -> RunManifest:
    if not run_dir.exists():
        raise RunNotFound(f"Run not found: {run_dir.name}")
    manifest_path = RunLayoutPaths(run_dir).manifest
    if not manifest_path.exists():
        raise UnsupportedRunSchema(UNSUPPORTED_V3_MESSAGE)
    payload = read_json(manifest_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise UnsupportedRunSchema(UNSUPPORTED_V3_MESSAGE)
    try:
        manifest = RunManifest.model_validate(payload)
    except ValidationError as exc:
        raise UnsupportedRunSchema(f"Unsupported run schema: {exc}") from exc
    return manifest.model_copy(
        update={
            "pages": [
                page.model_copy(update={"resolved_path": run_dir / page.image_path})
                for page in manifest.pages
            ]
        }
    )
