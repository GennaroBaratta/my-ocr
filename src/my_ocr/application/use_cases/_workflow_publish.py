from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil
import tempfile
from typing import Any

from ._workflow_payloads import (
    normalize_published_ocr_artifacts,
    normalize_published_review_json,
)


PathPair = tuple[Path, Path]


def create_staged_ocr_paths(paths: Any, run_paths_cls: type) -> Any:
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{paths.run_name}-ocr-", dir=str(paths.run_dir.parent))
    )
    return run_paths_cls.from_run_dir(staging_dir)


def create_staged_review_paths(paths: Any, run_paths_cls: type) -> Any:
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{paths.run_name}-review-", dir=str(paths.run_dir.parent))
    )
    return run_paths_cls.from_run_dir(staging_dir)


def publish_staged_ocr_run(
    source_paths: Any,
    target_paths: Any,
    *,
    run_paths_cls: type,
    include_pages: bool = True,
    post_publish: Callable[[], None] | None = None,
    move_path_fn: Callable[[Path, Path], None] = None,
    copy_path_fn: Callable[[Path, Path], None] = None,
    remove_paths_fn: Callable[[list[Path]], None] = None,
    normalize_ocr_fn: Callable[..., None] = normalize_published_ocr_artifacts,
) -> None:
    move_path_fn = move_path if move_path_fn is None else move_path_fn
    copy_path_fn = copy_path if copy_path_fn is None else copy_path_fn
    remove_paths_fn = remove_paths if remove_paths_fn is None else remove_paths_fn
    publish_staged_run(
        source_paths,
        target_paths,
        run_paths_cls=run_paths_cls,
        kind="ocr",
        move_artifacts=lambda source, target, moved_pairs: move_ocr_artifacts(
            source,
            target,
            moved_pairs=moved_pairs,
            include_pages=include_pages,
            move_path_fn=move_path_fn,
        ),
        normalize=lambda: normalize_ocr_fn(
            target_paths,
            source_run_dir=source_paths.run_dir,
            target_run_dir=target_paths.run_dir,
        ),
        restore_error="Failed to restore OCR artifacts after publish failure.",
        post_publish=post_publish,
        move_path_fn=move_path_fn,
        copy_path_fn=copy_path_fn,
        remove_paths_fn=remove_paths_fn,
    )


def publish_staged_review_run(
    source_paths: Any,
    target_paths: Any,
    *,
    run_paths_cls: type,
    post_publish: Callable[[], None] | None = None,
    move_path_fn: Callable[[Path, Path], None] = None,
    copy_path_fn: Callable[[Path, Path], None] = None,
    remove_paths_fn: Callable[[list[Path]], None] = None,
) -> None:
    move_path_fn = move_path if move_path_fn is None else move_path_fn
    copy_path_fn = copy_path if copy_path_fn is None else copy_path_fn
    remove_paths_fn = remove_paths if remove_paths_fn is None else remove_paths_fn
    publish_staged_run(
        source_paths,
        target_paths,
        run_paths_cls=run_paths_cls,
        kind="review",
        move_artifacts=lambda source, target, moved_pairs: move_review_artifacts(
            source,
            target,
            moved_pairs=moved_pairs,
            move_path_fn=move_path_fn,
        ),
        normalize=lambda: normalize_published_review_json(
            target_paths.reviewed_layout_path,
            source_run_dir=source_paths.run_dir,
            target_run_dir=target_paths.run_dir,
        ),
        restore_error="Failed to restore review artifacts after publish failure.",
        post_publish=post_publish,
        post_publish_cleanup=lambda: remove_replaced_ocr_outputs(target_paths),
        move_path_fn=move_path_fn,
        copy_path_fn=copy_path_fn,
        remove_paths_fn=remove_paths_fn,
    )


def publish_staged_run(
    source_paths: Any,
    target_paths: Any,
    *,
    run_paths_cls: type,
    kind: str,
    move_artifacts: Callable[[Any, Any, list[PathPair]], None],
    normalize: Callable[[], None],
    restore_error: str,
    post_publish: Callable[[], None] | None = None,
    post_publish_cleanup: Callable[[], None] | None = None,
    move_path_fn: Callable[[Path, Path], None] | None = None,
    copy_path_fn: Callable[[Path, Path], None] | None = None,
    remove_paths_fn: Callable[[list[Path]], None] | None = None,
) -> None:
    move_path_fn = move_path if move_path_fn is None else move_path_fn
    copy_path_fn = copy_path if copy_path_fn is None else copy_path_fn
    remove_paths_fn = remove_paths if remove_paths_fn is None else remove_paths_fn
    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_paths.run_name}-{kind}-backup-",
            dir=str(target_paths.run_dir.parent),
        )
    )
    backup_paths = run_paths_cls.from_run_dir(backup_dir)
    backed_up_pairs: list[PathPair] = []
    published_pairs: list[PathPair] = []
    try:
        if target_paths.meta_path.exists():
            move_path_fn(target_paths.meta_path, backup_paths.meta_path)
            backed_up_pairs.append((target_paths.meta_path, backup_paths.meta_path))
        move_artifacts(target_paths, backup_paths, backed_up_pairs)
        move_artifacts(source_paths, target_paths, published_pairs)
        normalize()
        if post_publish is not None:
            post_publish()
        if post_publish_cleanup is not None:
            post_publish_cleanup()
    except Exception:
        try:
            remove_paths_fn([target_paths.meta_path, *[target for _, target in published_pairs]])
            restore_moved_artifacts(backed_up_pairs, copy_path_fn=copy_path_fn)
        except Exception as restore_exc:
            raise RuntimeError(f"{restore_error} Backup preserved at {backup_dir}.") from restore_exc
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def move_ocr_artifacts(
    source_paths: Any,
    target_paths: Any,
    *,
    moved_pairs: list[PathPair] | None = None,
    include_pages: bool = True,
    move_path_fn: Callable[[Path, Path], None] | None = None,
) -> None:
    move_path_fn = move_path if move_path_fn is None else move_path_fn
    for source, target in ocr_artifact_pairs(
        source_paths,
        target_paths,
        include_pages=include_pages,
    ):
        if not source.exists():
            continue
        move_path_fn(source, target)
        if moved_pairs is not None:
            moved_pairs.append((source, target))


def move_review_artifacts(
    source_paths: Any,
    target_paths: Any,
    *,
    moved_pairs: list[PathPair] | None = None,
    move_path_fn: Callable[[Path, Path], None] | None = None,
) -> None:
    move_path_fn = move_path if move_path_fn is None else move_path_fn
    for source, target in review_artifact_pairs(source_paths, target_paths):
        if not source.exists():
            continue
        move_path_fn(source, target)
        if moved_pairs is not None:
            moved_pairs.append((source, target))


def restore_moved_artifacts(
    moved_pairs: list[PathPair],
    *,
    copy_path_fn: Callable[[Path, Path], None] | None = None,
) -> None:
    copy_path_fn = copy_path if copy_path_fn is None else copy_path_fn
    for source, target in reversed(moved_pairs):
        copy_path_fn(target, source)


def remove_ocr_artifacts(paths: Any) -> None:
    for target in paths.published_ocr_artifact_paths():
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()


def remove_replaced_ocr_outputs(paths: Any) -> None:
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


def remove_paths(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def ocr_artifact_pairs(
    source_paths: Any,
    target_paths: Any,
    *,
    include_pages: bool = True,
) -> list[PathPair]:
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
    return list(zip(source_artifacts, target_artifacts, strict=True))


def review_artifact_pairs(source_paths: Any, target_paths: Any) -> list[PathPair]:
    return list(
        zip(
            source_paths.published_review_artifact_paths(),
            target_paths.published_review_artifact_paths(),
            strict=True,
        )
    )


def move_path(source: Path, target: Path) -> None:
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)


def copy_path(source: Path, target: Path) -> None:
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
