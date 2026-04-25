from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_serializer, field_validator

SCHEMA_VERSION = 3


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
class LayoutOptions:
    config_path: str = "config/local.yaml"
    layout_device: str = "cuda"
    layout_profile: str | None = "auto"


@dataclass(frozen=True, slots=True)
class OcrOptions:
    config_path: str = "config/local.yaml"
    layout_device: str = "cuda"
    layout_profile: str | None = "auto"


@dataclass(frozen=True, slots=True)
class StructuredExtractionOptions:
    config_path: str = "config/local.yaml"
    model: str | None = None
    endpoint: str | None = None


class ApplicationError(Exception):
    """Base class for user-facing workflow failures."""


class RunNotFound(ApplicationError):
    pass


class UnsupportedRunSchema(ApplicationError):
    pass


class MissingInputDocument(ApplicationError):
    pass


class MissingPage(ApplicationError):
    pass


class LayoutDetectionFailed(ApplicationError):
    pass


class OcrFailed(ApplicationError):
    pass


class StructuredExtractionFailed(ApplicationError):
    pass


class RunCommitFailed(ApplicationError):
    pass


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class StrictModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class RunId(RootModel[str]):
    model_config = ConfigDict(frozen=True, strict=True)

    @field_validator("root")
    @classmethod
    def _validate_root(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run id cannot be empty")
        return value

    @field_serializer("root")
    def _serialize_root(self, value: str) -> str:
        return value

    def __str__(self) -> str:
        return self.root

    def __hash__(self) -> int:
        return hash(self.root)


class RunInput(StrictModel):
    path: str
    name: str
    kind: str

    @classmethod
    def from_path(cls, path: str | Path) -> "RunInput":
        source = Path(path)
        kind = "directory" if source.is_dir() else source.suffix.lower().lstrip(".") or "unknown"
        return cls(path=str(path), name=source.name, kind=kind)


class PageRef(StrictModel):
    page_number: int
    image_path: str
    width: int
    height: int
    resolved_path: Path | None = Field(default=None, exclude=True, repr=False)

    @property
    def path_for_io(self) -> Path:
        return self.resolved_path if self.resolved_path is not None else Path(self.image_path)


class LayoutDiagnostics(RootModel[dict[str, Any]]):
    model_config = ConfigDict(frozen=True, strict=True)

    def __init__(self, root: dict[str, Any] | None = None, **data: Any) -> None:
        super().__init__(data if root is None and data else root or {})

    @property
    def payload(self) -> dict[str, Any]:
        return self.root

    @property
    def warning(self) -> str | None:
        value = self.payload.get("layout_profile_warning")
        return value if isinstance(value, str) and value.strip() else None


class RunStatus(StrictModel):
    layout: str = "pending"
    ocr: str = "pending"
    extraction: str = "pending"


class RunDiagnostics(StrictModel):
    layout: LayoutDiagnostics = Field(default_factory=LayoutDiagnostics)
    ocr: dict[str, Any] = Field(default_factory=dict)
    extraction: dict[str, Any] = Field(default_factory=dict)


class RunManifest(StrictModel):
    run_id: RunId
    input: RunInput
    pages: list[PageRef]
    created_at: str
    updated_at: str
    status: RunStatus = Field(default_factory=RunStatus)
    diagnostics: RunDiagnostics = Field(default_factory=RunDiagnostics)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def new(cls, run_id: RunId, input_path: str | Path) -> "RunManifest":
        now = utc_now_iso()
        return cls(
            run_id=run_id,
            input=RunInput.from_path(input_path),
            pages=[],
            created_at=now,
            updated_at=now,
        )

    def with_updates(
        self,
        *,
        pages: list[PageRef] | None = None,
        status: RunStatus | None = None,
        diagnostics: RunDiagnostics | None = None,
    ) -> "RunManifest":
        return RunManifest(
            run_id=self.run_id,
            input=self.input,
            pages=self.pages if pages is None else pages,
            created_at=self.created_at,
            updated_at=utc_now_iso(),
            status=self.status if status is None else status,
            diagnostics=self.diagnostics if diagnostics is None else diagnostics,
            schema_version=self.schema_version,
        )


class LayoutBlock(StrictModel):
    id: str
    index: int
    label: str
    bbox: list[float]
    confidence: float = 1.0
    content: str = ""


class ReviewPage(StrictModel):
    page_number: int
    image_path: str
    image_width: int
    image_height: int
    blocks: list[LayoutBlock]
    provider_path: str | None = None
    coord_space: str = "pixel"


class ReviewLayout(StrictModel):
    pages: list[ReviewPage]
    status: str = "prepared"
    version: int = 3

    @property
    def summary(self) -> dict[str, int]:
        return {"page_count": len(self.pages)}


class OcrPageResult(StrictModel):
    page_number: int
    image_path: str
    markdown: str
    markdown_source: str = "unknown"
    provider_path: str | None = None
    fallback_path: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class OcrRunResult(StrictModel):
    pages: list[OcrPageResult]
    markdown: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        return {"page_count": len(self.pages), "sources": self.source_counts()}

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for page in self.pages:
            counts[page.markdown_source] = counts.get(page.markdown_source, 0) + 1
        return counts


class RunSnapshot(StrictModel):
    run_dir: Path
    manifest: RunManifest
    review_layout: ReviewLayout | None = None
    ocr_result: OcrRunResult | None = None
    extraction: dict[str, Any] = Field(default_factory=dict)

    @property
    def run_id(self) -> RunId:
        return self.manifest.run_id

    @property
    def pages(self) -> list[PageRef]:
        return self.manifest.pages

    def page(self, page_number: int) -> PageRef | None:
        for page in self.manifest.pages:
            if page.page_number == page_number:
                return page
        return None


@dataclass(frozen=True, slots=True)
class LayoutDetectionResult:
    layout: ReviewLayout
    artifacts: ProviderArtifacts = dataclass_field(default_factory=ProviderArtifacts.empty)
    diagnostics: LayoutDiagnostics = dataclass_field(default_factory=LayoutDiagnostics)


@dataclass(frozen=True, slots=True)
class OcrRecognitionResult:
    result: OcrRunResult
    artifacts: ProviderArtifacts = dataclass_field(default_factory=ProviderArtifacts.empty)


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    snapshot: RunSnapshot
    warning: str | None = None
