from __future__ import annotations

from dataclasses import replace
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from my_ocr.application.dto import (
    ArtifactCopy,
    LayoutDiagnostics,
    OcrPageResult,
    OcrRunResult,
    PageRef,
    ProviderArtifacts,
    ReviewLayout,
    ReviewPage,
    RunId,
    RunManifest,
    RunSnapshot,
    SCHEMA_VERSION,
)
from my_ocr.application.errors import (
    MissingPage,
    RunCommitFailed,
    RunNotFound,
    UnsupportedRunSchema,
)

DEFAULT_RUN_ROOT = "data/runs"


class FilesystemRunStore:
    def __init__(self, run_root: str | Path = DEFAULT_RUN_ROOT) -> None:
        self.run_root = Path(run_root)

    def create_run(self, input_path: str | Path, run_id: RunId | None = None) -> "RunTransaction":
        self.run_root.mkdir(parents=True, exist_ok=True)
        resolved_run_id = run_id or RunId(f"{_slugify(Path(input_path).stem)}-{_timestamp_id()}")
        target_dir = self.run_root / str(resolved_run_id)
        work_dir = Path(
            tempfile.mkdtemp(prefix=f".{resolved_run_id}-work-", dir=str(self.run_root))
        )
        manifest = RunManifest.new(resolved_run_id, input_path)
        _write_json(work_dir / "run.json", manifest.to_dict())
        return RunTransaction(self, target_dir, work_dir, is_new=not target_dir.exists())

    def open_run(self, run_id: RunId | str) -> RunSnapshot:
        return _load_snapshot(self._run_dir(run_id))

    def begin_update(self, run_id: RunId | str) -> "RunTransaction":
        target_dir = self._run_dir(run_id)
        if not target_dir.exists():
            raise RunNotFound(f"Run not found: {run_id}")
        _load_manifest(target_dir)
        self.run_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix=f".{target_dir.name}-work-", dir=str(self.run_root)))
        shutil.copytree(target_dir, work_dir, dirs_exist_ok=True)
        return RunTransaction(self, target_dir, work_dir, is_new=False)

    def _run_dir(self, run_id: RunId | str) -> Path:
        return self.run_root / str(run_id)


class RunTransaction:
    def __init__(
        self,
        store: FilesystemRunStore,
        target_dir: Path,
        work_dir: Path,
        *,
        is_new: bool,
    ) -> None:
        self._store = store
        self._target_dir = target_dir
        self.work_dir = work_dir
        self._is_new = is_new
        self._closed = False
        self.run_id = RunId(target_dir.name)

    @property
    def pages_dir(self) -> Path:
        return self.work_dir / "pages"

    def write_pages(self, pages: list[PageRef]) -> None:
        manifest = self._manifest()
        manifest = manifest.with_updates(pages=pages)
        self._write_manifest(manifest)

    def write_review_layout(
        self,
        layout: ReviewLayout,
        artifacts: ProviderArtifacts,
        diagnostics: LayoutDiagnostics | None = None,
    ) -> None:
        _write_json(self.work_dir / "layout" / "review.json", layout.to_dict())
        self._copy_artifacts(artifacts)
        _remove_path(self.work_dir / "ocr")
        _remove_path(self.work_dir / "extraction")
        manifest = self._manifest()
        self._write_manifest(
            manifest.with_updates(
                status=replace(
                    manifest.status,
                    layout=layout.status or "prepared",
                    ocr="pending",
                    extraction="pending",
                ),
                diagnostics=(
                    manifest.diagnostics
                    if diagnostics is None
                    else replace(manifest.diagnostics, layout=diagnostics)
                ),
            )
        )

    def write_ocr_result(self, result: OcrRunResult, artifacts: ProviderArtifacts) -> None:
        ocr_dir = self.work_dir / "ocr"
        _write_text(ocr_dir / "markdown.md", result.markdown)
        _write_json(ocr_dir / "pages.json", result.to_dict())
        self._copy_artifacts(artifacts)
        manifest = self._manifest()
        self._write_manifest(
            manifest.with_updates(
                status=replace(manifest.status, ocr="complete", extraction="pending"),
                diagnostics=replace(manifest.diagnostics, ocr=dict(result.diagnostics)),
            )
        )
        self.clear_extraction_outputs()

    def replace_page_layout(
        self, page_number: int, page: ReviewPage, artifacts: ProviderArtifacts
    ) -> None:
        snapshot = _load_snapshot(self.work_dir)
        existing = snapshot.review_layout or ReviewLayout(pages=[], status="prepared")
        pages_by_number = {review_page.page_number: review_page for review_page in existing.pages}
        pages_by_number[page_number] = page
        ordered_pages = [
            pages_by_number[manifest_page.page_number]
            for manifest_page in snapshot.pages
            if manifest_page.page_number in pages_by_number
        ]
        self.write_review_layout(
            ReviewLayout(pages=ordered_pages, status="prepared", version=existing.version),
            artifacts,
        )

    def replace_page_ocr(
        self, page_number: int, page: OcrPageResult, artifacts: ProviderArtifacts
    ) -> None:
        snapshot = _load_snapshot(self.work_dir)
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
        self.write_ocr_result(
            OcrRunResult(
                pages=ordered_pages,
                markdown=markdown,
                diagnostics=snapshot.ocr_result.diagnostics,
            ),
            artifacts,
        )

    def write_rules_extraction(self, prediction: dict[str, Any]) -> None:
        _write_json(self.work_dir / "extraction" / "rules.json", prediction)
        _write_json(self.work_dir / "extraction" / "canonical.json", prediction)
        manifest = self._manifest()
        self._write_manifest(
            manifest.with_updates(status=replace(manifest.status, extraction="rules"))
        )

    def write_structured_extraction(
        self,
        prediction: dict[str, Any],
        metadata: dict[str, Any],
        *,
        canonical_prediction: dict[str, Any],
    ) -> None:
        _write_json(self.work_dir / "extraction" / "structured.json", prediction)
        _write_json(self.work_dir / "extraction" / "structured_meta.json", metadata)
        _write_json(self.work_dir / "extraction" / "canonical.json", canonical_prediction)
        manifest = self._manifest()
        self._write_manifest(
            manifest.with_updates(
                status=replace(manifest.status, extraction="structured"),
                diagnostics=replace(manifest.diagnostics, extraction=dict(metadata)),
            )
        )

    def clear_extraction_outputs(self) -> None:
        extraction_dir = self.work_dir / "extraction"
        if extraction_dir.exists():
            shutil.rmtree(extraction_dir)
        manifest = self._manifest()
        if manifest.status.extraction != "pending":
            self._write_manifest(
                manifest.with_updates(status=replace(manifest.status, extraction="pending"))
            )

    def commit(self) -> RunSnapshot:
        if self._closed:
            return _load_snapshot(self._target_dir)
        _load_manifest(self.work_dir)
        backup_dir: Path | None = None
        try:
            self._target_dir.parent.mkdir(parents=True, exist_ok=True)
            if self._target_dir.exists():
                backup_dir = Path(
                    tempfile.mkdtemp(
                        prefix=f".{self._target_dir.name}-backup-",
                        dir=str(self._target_dir.parent),
                    )
                )
                shutil.rmtree(backup_dir)
                self._target_dir.replace(backup_dir)
            self.work_dir.replace(self._target_dir)
        except Exception as exc:
            if self._target_dir.exists():
                shutil.rmtree(self._target_dir, ignore_errors=True)
            if backup_dir is not None and backup_dir.exists():
                backup_dir.replace(self._target_dir)
            raise RunCommitFailed(f"Failed to commit run {self.run_id}") from exc
        finally:
            if backup_dir is not None and backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            self._closed = True
        return _load_snapshot(self._target_dir)

    def rollback(self) -> None:
        if not self._closed:
            shutil.rmtree(self.work_dir, ignore_errors=True)
            self._closed = True

    def _manifest(self) -> RunManifest:
        return _load_manifest(self.work_dir)

    def _write_manifest(self, manifest: RunManifest) -> None:
        _write_json(self.work_dir / "run.json", manifest.to_dict())

    def _copy_artifacts(self, artifacts: ProviderArtifacts) -> None:
        try:
            for item in artifacts.copies:
                _copy_path(item, self.work_dir)
        finally:
            for path in artifacts.cleanup_paths:
                shutil.rmtree(path, ignore_errors=True)


def _copy_path(item: ArtifactCopy, work_dir: Path) -> None:
    target = item.relative_target
    if Path(target).is_absolute() or ".." in Path(target).parts:
        raise ValueError(f"Artifact target must be run-relative: {target}")
    destination = work_dir / target
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
            "Unsupported run schema: this run was created before v2. "
            "Re-run the document to create a v2 run."
        )
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise UnsupportedRunSchema(
            "Unsupported run schema: this run was created before v2. "
            "Re-run the document to create a v2 run."
        )
    return RunManifest.from_dict(payload, run_dir=run_dir)


def _load_review_layout(run_dir: Path) -> ReviewLayout | None:
    path = run_dir / "layout" / "review.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    return ReviewLayout.from_dict(payload) if isinstance(payload, dict) else None


def _load_ocr_result(run_dir: Path) -> OcrRunResult | None:
    pages_path = run_dir / "ocr" / "pages.json"
    markdown_path = run_dir / "ocr" / "markdown.md"
    if not pages_path.exists():
        return None
    payload = _read_json(pages_path)
    markdown = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
    return OcrRunResult.from_dict(payload, markdown=markdown) if isinstance(payload, dict) else None


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
