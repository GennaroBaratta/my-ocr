from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from my_ocr.filesystem import read_json as _read_json
from my_ocr.filesystem import read_text as _read_text
from my_ocr.filesystem import write_json as _write_json
from my_ocr.filesystem import write_text as _write_text
from my_ocr.models import (
    ArtifactCopy,
    LayoutDiagnostics,
    MissingPage,
    OcrPageResult,
    OcrRunResult,
    PageRef,
    ProviderArtifacts,
    ReviewLayout,
    ReviewPage,
    RunCommitFailed,
    RunDiagnostics,
    RunId,
    RunManifest,
    RunNotFound,
    RunSnapshot,
    RunStatus,
    SCHEMA_VERSION,
    UnsupportedRunSchema,
)

DEFAULT_RUN_ROOT = "data/runs"


@dataclass(frozen=True, slots=True)
class RunWorkspace:
    run_id: RunId
    target_dir: Path
    work_dir: Path


class RunStorage(Protocol):
    def start_run(
        self, input_path: str | Path, run_id: RunId | None = None
    ) -> RunWorkspace: ...

    def publish_prepared_run(
        self,
        workspace: RunWorkspace,
        pages: list[PageRef],
        layout: ReviewLayout,
        artifacts: ProviderArtifacts,
        diagnostics: LayoutDiagnostics | None = None,
    ) -> RunSnapshot: ...

    def discard_workspace(self, workspace: RunWorkspace) -> None: ...

    def open_run(self, run_id: RunId | str) -> RunSnapshot: ...

    def write_review_layout(
        self,
        run_id: RunId | str,
        layout: ReviewLayout,
        artifacts: ProviderArtifacts,
        diagnostics: LayoutDiagnostics | None = None,
    ) -> RunSnapshot: ...

    def write_ocr_result(
        self,
        run_id: RunId | str,
        result: OcrRunResult,
        artifacts: ProviderArtifacts,
    ) -> RunSnapshot: ...

    def replace_page_layout(
        self,
        run_id: RunId | str,
        page_number: int,
        page: ReviewPage,
        artifacts: ProviderArtifacts,
    ) -> RunSnapshot: ...

    def replace_page_ocr(
        self,
        run_id: RunId | str,
        page_number: int,
        page: OcrPageResult,
        artifacts: ProviderArtifacts,
    ) -> RunSnapshot: ...

    def write_rules_extraction(
        self, run_id: RunId | str, prediction: dict[str, Any]
    ) -> RunSnapshot: ...

    def write_structured_extraction(
        self,
        run_id: RunId | str,
        prediction: dict[str, Any],
        metadata: dict[str, Any],
        *,
        canonical_prediction: dict[str, Any],
    ) -> RunSnapshot: ...

    def clear_extraction_outputs(self, run_id: RunId | str) -> RunSnapshot: ...


class FilesystemRunStore:
    def __init__(self, run_root: str | Path = DEFAULT_RUN_ROOT) -> None:
        self.run_root = Path(run_root)

    def start_run(self, input_path: str | Path, run_id: RunId | None = None) -> RunWorkspace:
        self.run_root.mkdir(parents=True, exist_ok=True)
        resolved_run_id = run_id or RunId(f"{_slugify(Path(input_path).stem)}-{_timestamp_id()}")
        target_dir = self._run_dir(resolved_run_id)
        work_dir = Path(
            tempfile.mkdtemp(prefix=f".{resolved_run_id}-work-", dir=str(self.run_root))
        )
        _write_manifest(work_dir, RunManifest.new(resolved_run_id, input_path))
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
            manifest = _load_manifest(workspace.work_dir)
            _write_manifest(
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

    def write_review_layout(
        self,
        run_id: RunId | str,
        layout: ReviewLayout,
        artifacts: ProviderArtifacts,
        diagnostics: LayoutDiagnostics | None = None,
    ) -> RunSnapshot:
        run_dir = self._existing_run_dir(run_id)
        snapshot = _load_snapshot(run_dir)
        _write_review_layout_to_dir(run_dir, layout, artifacts)
        _remove_path(run_dir / "ocr")
        _remove_path(run_dir / "extraction")
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
        _write_manifest(run_dir, manifest)
        return _load_snapshot(run_dir)

    def write_ocr_result(
        self, run_id: RunId | str, result: OcrRunResult, artifacts: ProviderArtifacts
    ) -> RunSnapshot:
        run_dir = self._existing_run_dir(run_id)
        snapshot = _load_snapshot(run_dir)
        _write_ocr_result_to_dir(run_dir, result, artifacts)
        _remove_path(run_dir / "extraction")
        _write_manifest(
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

    def replace_page_layout(
        self,
        run_id: RunId | str,
        page_number: int,
        page: ReviewPage,
        artifacts: ProviderArtifacts,
    ) -> RunSnapshot:
        snapshot = _load_snapshot(self._existing_run_dir(run_id))
        existing = snapshot.review_layout or ReviewLayout(pages=[], status="prepared")
        pages_by_number = {review_page.page_number: review_page for review_page in existing.pages}
        pages_by_number[page_number] = page
        ordered_pages = [
            pages_by_number[manifest_page.page_number]
            for manifest_page in snapshot.pages
            if manifest_page.page_number in pages_by_number
        ]
        return self.write_review_layout(
            run_id,
            ReviewLayout(pages=ordered_pages, status="prepared", version=existing.version),
            artifacts,
        )

    def replace_page_ocr(
        self,
        run_id: RunId | str,
        page_number: int,
        page: OcrPageResult,
        artifacts: ProviderArtifacts,
    ) -> RunSnapshot:
        snapshot = _load_snapshot(self._existing_run_dir(run_id))
        if snapshot.ocr_result is None:
            raise MissingPage("Cannot replace a page before OCR has been run.")
        pages_by_number = {
            ocr_page.page_number: ocr_page for ocr_page in snapshot.ocr_result.pages
        }
        pages_by_number[page_number] = page
        ordered_pages = [
            pages_by_number[manifest_page.page_number]
            for manifest_page in snapshot.pages
            if manifest_page.page_number in pages_by_number
        ]
        markdown = "\n\n".join(page.markdown.strip() for page in ordered_pages if page.markdown.strip())
        return self.write_ocr_result(
            run_id,
            OcrRunResult(
                pages=ordered_pages,
                markdown=markdown,
                diagnostics=snapshot.ocr_result.diagnostics,
            ),
            artifacts,
        )

    def write_rules_extraction(
        self, run_id: RunId | str, prediction: dict[str, Any]
    ) -> RunSnapshot:
        run_dir = self._existing_run_dir(run_id)
        snapshot = _load_snapshot(run_dir)
        _write_json(run_dir / "extraction" / "rules.json", prediction)
        _write_json(run_dir / "extraction" / "canonical.json", prediction)
        _write_manifest(
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
        _write_json(run_dir / "extraction" / "structured.json", prediction)
        _write_json(run_dir / "extraction" / "structured_meta.json", metadata)
        _write_json(run_dir / "extraction" / "canonical.json", canonical_prediction)
        _write_manifest(
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
                    extraction=dict(metadata),
                ),
            ),
        )
        return _load_snapshot(run_dir)

    def clear_extraction_outputs(self, run_id: RunId | str) -> RunSnapshot:
        run_dir = self._existing_run_dir(run_id)
        snapshot = _load_snapshot(run_dir)
        _remove_path(run_dir / "extraction")
        if snapshot.manifest.status.extraction != "pending":
            _write_manifest(
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

    def _existing_run_dir(self, run_id: RunId | str) -> Path:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise RunNotFound(f"Run not found: {run_id}")
        _load_manifest(run_dir)
        return run_dir


def _publish_workspace(workspace: RunWorkspace) -> RunSnapshot:
    _load_manifest(workspace.work_dir)
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


def _write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    _write_json(run_dir / "run.json", manifest.model_dump(mode="json"))


def _write_review_layout_to_dir(
    run_dir: Path, layout: ReviewLayout, artifacts: ProviderArtifacts
) -> None:
    _write_json(run_dir / "layout" / "review.json", layout.model_dump(mode="json"))
    _copy_artifacts(artifacts, run_dir)


def _write_ocr_result_to_dir(
    run_dir: Path, result: OcrRunResult, artifacts: ProviderArtifacts
) -> None:
    _write_text(run_dir / "ocr" / "markdown.md", result.markdown)
    _write_json(run_dir / "ocr" / "pages.json", result.model_dump(mode="json"))
    _copy_artifacts(artifacts, run_dir)


def _copy_artifacts(artifacts: ProviderArtifacts, run_dir: Path) -> None:
    try:
        for item in artifacts.copies:
            _copy_path(item, run_dir)
    finally:
        for path in artifacts.cleanup_paths:
            _remove_path(path)


def _copy_path(item: ArtifactCopy, run_dir: Path) -> None:
    target = Path(item.relative_target)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(f"Artifact target must be run-relative: {target}")
    destination = run_dir / target
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if item.source.is_dir():
        shutil.copytree(item.source, destination)
    elif item.source.exists():
        shutil.copy2(item.source, destination)


def _load_snapshot(run_dir: Path) -> RunSnapshot:
    manifest = _load_manifest(run_dir)
    review_layout = _load_review_layout(run_dir)
    ocr_result = _load_ocr_result(run_dir)
    extraction = _load_extraction(run_dir)
    return RunSnapshot(
        run_dir=run_dir,
        manifest=manifest,
        review_layout=review_layout,
        ocr_result=ocr_result,
        extraction=extraction,
    )


def _load_manifest(run_dir: Path) -> RunManifest:
    if not run_dir.exists():
        raise RunNotFound(f"Run not found: {run_dir.name}")
    manifest_path = run_dir / "run.json"
    if not manifest_path.exists():
        raise UnsupportedRunSchema(
            "Unsupported run schema: this run was created before v3. "
            "Re-run the document to create a v3 run."
        )
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise UnsupportedRunSchema(
            "Unsupported run schema: this run was created before v3. "
            "Re-run the document to create a v3 run."
        )
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


def _load_review_layout(run_dir: Path) -> ReviewLayout | None:
    path = run_dir / "layout" / "review.json"
    if not path.exists():
        return None
    return ReviewLayout.model_validate(_read_json(path))


def _load_ocr_result(run_dir: Path) -> OcrRunResult | None:
    pages_path = run_dir / "ocr" / "pages.json"
    markdown_path = run_dir / "ocr" / "markdown.md"
    if not pages_path.exists():
        return None
    payload = _read_json(pages_path)
    if not isinstance(payload, dict):
        return None
    if "markdown" not in payload and markdown_path.exists():
        payload = {**payload, "markdown": _read_text(markdown_path)}
    return OcrRunResult.model_validate(payload)


def _load_extraction(run_dir: Path) -> dict[str, Any]:
    extraction_dir = run_dir / "extraction"
    if not extraction_dir.exists():
        return {}
    payload: dict[str, Any] = {}
    for key, filename in (
        ("rules", "rules.json"),
        ("structured", "structured.json"),
        ("structured_meta", "structured_meta.json"),
        ("canonical", "canonical.json"),
    ):
        path = extraction_dir / filename
        if path.exists():
            payload[key] = _read_json(path)
    return payload


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


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
    unsupported: bool = False


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
            try:
                snapshot = self.store.open_run(run_dir.name)
            except UnsupportedRunSchema:
                records.append(
                    RecentRunRecord(
                        run_id=run_dir.name,
                        input_path="",
                        mtime=run_dir.stat().st_mtime,
                        status="unsupported",
                        unsupported=True,
                    )
                )
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
