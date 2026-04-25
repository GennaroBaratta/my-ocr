from __future__ import annotations

import shutil
from pathlib import Path

from my_ocr.models import ArtifactCopy, ProviderArtifacts


def copy_provider_artifacts(artifacts: ProviderArtifacts, run_dir: Path) -> None:
    try:
        for item in artifacts.copies:
            copy_artifact(item, run_dir)
    finally:
        cleanup_artifacts(artifacts.cleanup_paths)


def cleanup_artifacts(paths: tuple[Path, ...]) -> None:
    for path in paths:
        remove_path(path)


def copy_artifact(item: ArtifactCopy, run_dir: Path) -> None:
    target = Path(item.relative_target)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(f"Artifact target must be run-relative: {target}")
    destination = run_dir / target
    remove_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if item.source.is_dir():
        shutil.copytree(item.source, destination)
    elif item.source.exists():
        shutil.copy2(item.source, destination)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
