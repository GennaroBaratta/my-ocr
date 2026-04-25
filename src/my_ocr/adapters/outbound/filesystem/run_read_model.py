from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from my_ocr.application.models import RunSnapshot
from my_ocr.application.errors import UnsupportedRunSchema

from .run_store import DEFAULT_RUN_ROOT, FilesystemRunStore


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
