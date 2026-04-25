from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from my_ocr.application.models import LayoutDiagnostics, OcrRunResult, ReviewLayout


@dataclass(frozen=True, slots=True)
class ArtifactCopy:
    source: Path
    relative_target: str


@dataclass(frozen=True, slots=True)
class ProviderArtifacts:
    copies: tuple[ArtifactCopy, ...] = ()
    cleanup_paths: tuple[Path, ...] = ()

    @classmethod
    def empty(cls) -> "ProviderArtifacts":
        return cls(())


@dataclass(frozen=True, slots=True)
class LayoutDetectionResult:
    layout: ReviewLayout
    artifacts: ProviderArtifacts = field(default_factory=ProviderArtifacts.empty)
    diagnostics: LayoutDiagnostics = field(default_factory=LayoutDiagnostics)


@dataclass(frozen=True, slots=True)
class OcrRecognitionResult:
    result: OcrRunResult
    artifacts: ProviderArtifacts = field(default_factory=ProviderArtifacts.empty)
