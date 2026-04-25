from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
