from __future__ import annotations

from dataclasses import dataclass
import re
import secrets
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from my_ocr.runs.artifacts import copy_provider_artifacts
from my_ocr.runs.artifacts import remove_path
from my_ocr.support.filesystem import write_json as _write_json
from my_ocr.domain import (
    LayoutDiagnostics,
    OcrRunResult,
    PageRef,
    ProviderArtifacts,
    ReviewLayout,
    RunCommitFailed,
    RunDiagnostics,
    RunId,
    RunManifest,
    RunNotFound,
    RunSnapshot,
    RunStatus,
)
from my_ocr.runs.layout import (
    RunLayoutPaths,
    load_extraction,
    load_ocr_result,
    load_review_layout,
    write_ocr_result_payload,
    write_review_layout_payload,
)
from my_ocr.runs.manifest import load_manifest, write_manifest

DEFAULT_RUN_ROOT = "data/runs"
STRUCTURED_RAW_BODY_METADATA_KEY = "_raw_body"


@dataclass(frozen=True, slots=True)
class RunWorkspace:
    run_id: RunId
    target_dir: Path
    work_dir: Path


class FilesystemRunStore:
    def __init__(self, run_root: str | Path = DEFAULT_RUN_ROOT) -> None:
        self.run_root = Path(run_root)

    def start_run(self, input_path: str | Path, run_id: RunId | None = None) -> RunWorkspace:
        self.run_root.mkdir(parents=True, exist_ok=True)
        resolved_run_id = run_id or self._new_run_id(input_path)
        target_dir = self._run_dir(resolved_run_id)
        work_dir = Path(
            tempfile.mkdtemp(prefix=f".{resolved_run_id}-work-", dir=str(self.run_root))
        )
        write_manifest(work_dir, RunManifest.new(resolved_run_id, input_path))
        return RunWorkspace(resolved_run_id, target_dir, work_dir)

    def publish_prepared_run(
        self,
        workspace: RunWorkspace,
        pages: list[PageRef],
        layout: ReviewLayout,
        artifacts: ProviderArtifacts,
        diagnostics: LayoutDiagnostics | None = None,
    ) -> RunSnapshot:
        try:
            manifest = load_manifest(workspace.work_dir)
            write_manifest(
                workspace.work_dir,
                manifest.with_updates(
                    pages=pages,
                    status=RunStatus(layout=layout.status or "prepared"),
                    diagnostics=(
                        manifest.diagnostics
                        if diagnostics is None
                        else RunDiagnostics(layout=diagnostics)
                    ),
                ),
            )
            _write_review_layout_to_dir(workspace.work_dir, layout, artifacts)
            return _publish_workspace(workspace)
        except Exception:
            self.discard_workspace(workspace)
            raise

    def discard_workspace(self, workspace: RunWorkspace) -> None:
        shutil.rmtree(workspace.work_dir, ignore_errors=True)

    def open_run(self, run_id: RunId | str) -> RunSnapshot:
        return _load_snapshot(self._run_dir(run_id))

    def save_review_layout_and_invalidate_downstream(
        self,
        run_id: RunId | str,
        layout: ReviewLayout,
        artifacts: ProviderArtifacts,
        diagnostics: LayoutDiagnostics | None = None,
    ) -> RunSnapshot:
        run_dir = self._existing_run_dir(run_id)
        snapshot = _load_snapshot(run_dir)
        _write_review_layout_to_dir(run_dir, layout, artifacts)
        paths = RunLayoutPaths(run_dir)
        remove_path(paths.ocr_dir)
        remove_path(paths.extraction_dir)
        manifest = snapshot.manifest.with_updates(
            status=RunStatus(layout=layout.status or "prepared", ocr="pending", extraction="pending"),
            diagnostics=(
                snapshot.manifest.diagnostics
                if diagnostics is None
                else RunDiagnostics(
                    layout=diagnostics,
                    ocr=snapshot.manifest.diagnostics.ocr,
                    extraction=snapshot.manifest.diagnostics.extraction,
                )
            ),
        )
        write_manifest(run_dir, manifest)
        return _load_snapshot(run_dir)

    def write_ocr_result_and_invalidate_extraction(
        self, run_id: RunId | str, result: OcrRunResult, artifacts: ProviderArtifacts
    ) -> RunSnapshot:
        run_dir = self._existing_run_dir(run_id)
        snapshot = _load_snapshot(run_dir)
        _write_ocr_result_to_dir(run_dir, result, artifacts)
        remove_path(RunLayoutPaths(run_dir).extraction_dir)
        write_manifest(
            run_dir,
            snapshot.manifest.with_updates(
                status=RunStatus(
                    layout=snapshot.manifest.status.layout,
                    ocr="complete",
                    extraction="pending",
                ),
                diagnostics=RunDiagnostics(
                    layout=snapshot.manifest.diagnostics.layout,
                    ocr=dict(result.diagnostics),
                    extraction={},
                ),
            ),
        )
        return _load_snapshot(run_dir)

    def write_rules_extraction(
        self, run_id: RunId | str, prediction: dict[str, Any]
    ) -> RunSnapshot:
        run_dir = self._existing_run_dir(run_id)
        snapshot = _load_snapshot(run_dir)
        paths = RunLayoutPaths(run_dir)
        _write_json(paths.rules_extraction, prediction)
        _write_json(paths.canonical_extraction, prediction)
        write_manifest(
            run_dir,
            snapshot.manifest.with_updates(
                status=RunStatus(
                    layout=snapshot.manifest.status.layout,
                    ocr=snapshot.manifest.status.ocr,
                    extraction="rules",
                )
            ),
        )
        return _load_snapshot(run_dir)

    def write_structured_extraction(
        self,
        run_id: RunId | str,
        prediction: dict[str, Any],
        metadata: dict[str, Any],
        *,
        canonical_prediction: dict[str, Any],
    ) -> RunSnapshot:
        run_dir = self._existing_run_dir(run_id)
        snapshot = _load_snapshot(run_dir)
        paths = RunLayoutPaths(run_dir)
        metadata_payload = dict(metadata)
        raw_body = metadata_payload.pop(STRUCTURED_RAW_BODY_METADATA_KEY, None)
        _write_json(paths.structured_extraction, prediction)
        _write_json(paths.structured_extraction_meta, metadata_payload)
        if raw_body is not None:
            _write_json(paths.structured_extraction_raw, raw_body)
        _write_json(paths.canonical_extraction, canonical_prediction)
        write_manifest(
            run_dir,
            snapshot.manifest.with_updates(
                status=RunStatus(
                    layout=snapshot.manifest.status.layout,
                    ocr=snapshot.manifest.status.ocr,
                    extraction="structured",
                ),
                diagnostics=RunDiagnostics(
                    layout=snapshot.manifest.diagnostics.layout,
                    ocr=snapshot.manifest.diagnostics.ocr,
                    extraction=metadata_payload,
                ),
            ),
        )
        return _load_snapshot(run_dir)

    def clear_extraction_outputs(self, run_id: RunId | str) -> RunSnapshot:
        run_dir = self._existing_run_dir(run_id)
        snapshot = _load_snapshot(run_dir)
        remove_path(RunLayoutPaths(run_dir).extraction_dir)
        if snapshot.manifest.status.extraction != "pending":
            write_manifest(
                run_dir,
                snapshot.manifest.with_updates(
                    status=RunStatus(
                        layout=snapshot.manifest.status.layout,
                        ocr=snapshot.manifest.status.ocr,
                        extraction="pending",
                    )
                ),
            )
        return _load_snapshot(run_dir)

    def _run_dir(self, run_id: RunId | str) -> Path:
        return self.run_root / str(run_id)

    def _new_run_id(self, input_path: str | Path) -> RunId:
        stem = _slugify(Path(input_path).stem)
        for _attempt in range(100):
            candidate = RunId(f"{stem}-{_timestamp_id()}-{secrets.token_hex(3)}")
            if not self._run_dir(candidate).exists():
                return candidate
        raise RunCommitFailed(f"Could not allocate a unique run id for {input_path}")

    def _existing_run_dir(self, run_id: RunId | str) -> Path:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise RunNotFound(f"Run not found: {run_id}")
        load_manifest(run_dir)
        return run_dir


def _publish_workspace(workspace: RunWorkspace) -> RunSnapshot:
    load_manifest(workspace.work_dir)
    backup_dir: Path | None = None
    try:
        workspace.target_dir.parent.mkdir(parents=True, exist_ok=True)
        if workspace.target_dir.exists():
            backup_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{workspace.target_dir.name}-backup-",
                    dir=str(workspace.target_dir.parent),
                )
            )
            shutil.rmtree(backup_dir)
            workspace.target_dir.replace(backup_dir)
        workspace.work_dir.replace(workspace.target_dir)
    except Exception as exc:
        if workspace.target_dir.exists():
            shutil.rmtree(workspace.target_dir, ignore_errors=True)
        if backup_dir is not None and backup_dir.exists():
            backup_dir.replace(workspace.target_dir)
        raise RunCommitFailed(f"Failed to publish run {workspace.run_id}") from exc
    finally:
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
    return _load_snapshot(workspace.target_dir)


def _write_review_layout_to_dir(
    run_dir: Path, layout: ReviewLayout, artifacts: ProviderArtifacts
) -> None:
    write_review_layout_payload(run_dir, layout)
    copy_provider_artifacts(artifacts, run_dir)


def _write_ocr_result_to_dir(
    run_dir: Path, result: OcrRunResult, artifacts: ProviderArtifacts
) -> None:
    write_ocr_result_payload(run_dir, result)
    copy_provider_artifacts(artifacts, run_dir)


def _load_snapshot(run_dir: Path) -> RunSnapshot:
    manifest = load_manifest(run_dir)
    review_layout = load_review_layout(run_dir)
    ocr_result = load_ocr_result(run_dir)
    extraction = load_extraction(run_dir)
    return RunSnapshot(
        run_dir=run_dir,
        manifest=manifest,
        review_layout=review_layout,
        ocr_result=ocr_result,
        extraction=extraction,
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def _timestamp_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True, slots=True)
class RecentRunRecord:
    run_id: str
    input_path: str
    mtime: float
    status: str


class FilesystemRunReadModel:
    def __init__(self, run_root: str | Path = DEFAULT_RUN_ROOT) -> None:
        self.run_root = Path(run_root)
        self.store = FilesystemRunStore(self.run_root)

    def list_recent_runs(self) -> list[RecentRunRecord]:
        if not self.run_root.exists():
            return []
        records: list[RecentRunRecord] = []
        for run_dir in sorted(
            (path for path in self.run_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            if run_dir.name.startswith("."):
                continue
            if not RunLayoutPaths(run_dir).manifest.exists():
                continue
            try:
                snapshot = self.store.open_run(run_dir.name)
            except Exception:
                continue
            records.append(
                RecentRunRecord(
                    run_id=str(snapshot.run_id),
                    input_path=snapshot.manifest.input.path,
                    mtime=run_dir.stat().st_mtime,
                    status=_status_for_snapshot(snapshot),
                )
            )
        return records

    def load_run(self, run_id: str) -> RunSnapshot:
        return self.store.open_run(run_id)


def _status_for_snapshot(snapshot: RunSnapshot) -> str:
    if snapshot.manifest.status.extraction != "pending":
        return "extracted"
    if snapshot.manifest.status.ocr == "complete":
        return "ocr_complete"
    if snapshot.review_layout is not None:
        return "review_ready"
    return "pending"
