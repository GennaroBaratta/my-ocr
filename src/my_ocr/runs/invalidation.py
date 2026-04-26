from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from my_ocr.domain import LayoutDiagnostics, RunDiagnostics, RunManifest, RunStatus

InvalidatedArtifactGroup = Literal["ocr", "extraction"]
LayoutStatus = Literal["prepared", "reviewed"]


@dataclass(frozen=True, slots=True)
class RunInvalidationPlan:
    artifact_groups: tuple[InvalidatedArtifactGroup, ...]
    status: RunStatus | None = None
    diagnostics: RunDiagnostics | None = None

    @property
    def updates_manifest(self) -> bool:
        return self.status is not None or self.diagnostics is not None


def review_layout_updated_policy(
    manifest: RunManifest,
    *,
    layout_status: LayoutStatus | None,
    diagnostics: LayoutDiagnostics | None = None,
) -> RunInvalidationPlan:
    resolved_diagnostics = (
        manifest.diagnostics
        if diagnostics is None
        else RunDiagnostics(
            layout=diagnostics,
            ocr=manifest.diagnostics.ocr,
            extraction=manifest.diagnostics.extraction,
        )
    )
    return RunInvalidationPlan(
        artifact_groups=("ocr", "extraction"),
        status=RunStatus(
            layout=layout_status or "prepared",
            ocr="pending",
            extraction="pending",
        ),
        diagnostics=resolved_diagnostics,
    )


def ocr_result_updated_policy(
    manifest: RunManifest,
    *,
    diagnostics: Mapping[str, object],
) -> RunInvalidationPlan:
    return RunInvalidationPlan(
        artifact_groups=("extraction",),
        status=RunStatus(
            layout=manifest.status.layout,
            ocr="complete",
            extraction="pending",
        ),
        diagnostics=RunDiagnostics(
            layout=manifest.diagnostics.layout,
            ocr=dict(diagnostics),
            extraction={},
        ),
    )


def extraction_outputs_cleared_policy(manifest: RunManifest) -> RunInvalidationPlan:
    if manifest.status.extraction == "pending":
        return RunInvalidationPlan(artifact_groups=("extraction",))
    return RunInvalidationPlan(
        artifact_groups=("extraction",),
        status=RunStatus(
            layout=manifest.status.layout,
            ocr=manifest.status.ocr,
            extraction="pending",
        ),
    )


__all__ = [
    "InvalidatedArtifactGroup",
    "RunInvalidationPlan",
    "extraction_outputs_cleared_policy",
    "ocr_result_updated_policy",
    "review_layout_updated_policy",
]
