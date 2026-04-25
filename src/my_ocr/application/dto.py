from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class RunId:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("run id cannot be empty")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class RunInput:
    path: str
    name: str
    kind: str

    @classmethod
    def from_path(cls, path: str | Path) -> "RunInput":
        source = Path(path)
        kind = "directory" if source.is_dir() else source.suffix.lower().lstrip(".") or "unknown"
        return cls(path=str(path), name=source.name, kind=kind)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "name": self.name, "kind": self.kind}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunInput":
        return cls(
            path=str(payload.get("path", "")),
            name=str(payload.get("name", "")),
            kind=str(payload.get("kind", "unknown")),
        )


@dataclass(frozen=True, slots=True)
class PageRef:
    page_number: int
    image_path: str
    width: int
    height: int
    resolved_path: Path | None = field(default=None, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, run_dir: Path | None = None) -> "PageRef":
        image_path = str(payload.get("image_path", ""))
        return cls(
            page_number=int(payload.get("page_number", 0)),
            image_path=image_path,
            width=int(payload.get("width", 0)),
            height=int(payload.get("height", 0)),
            resolved_path=(run_dir / image_path if run_dir is not None and image_path else None),
        )

    @property
    def path_for_io(self) -> Path:
        if self.resolved_path is None:
            return Path(self.image_path)
        return self.resolved_path


@dataclass(frozen=True, slots=True)
class LayoutDiagnostics:
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def warning(self) -> str | None:
        value = self.payload.get("layout_profile_warning")
        return value if isinstance(value, str) and value.strip() else None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @classmethod
    def from_dict(cls, payload: Any) -> "LayoutDiagnostics":
        return cls(dict(payload) if isinstance(payload, dict) else {})


@dataclass(frozen=True, slots=True)
class RunStatus:
    layout: str = "pending"
    ocr: str = "pending"
    extraction: str = "pending"

    def to_dict(self) -> dict[str, str]:
        return {"layout": self.layout, "ocr": self.ocr, "extraction": self.extraction}

    @classmethod
    def from_dict(cls, payload: Any) -> "RunStatus":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            layout=str(payload.get("layout", "pending")),
            ocr=str(payload.get("ocr", "pending")),
            extraction=str(payload.get("extraction", "pending")),
        )


@dataclass(frozen=True, slots=True)
class RunDiagnostics:
    layout: LayoutDiagnostics = field(default_factory=LayoutDiagnostics)
    ocr: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout.to_dict(),
            "ocr": dict(self.ocr),
            "extraction": dict(self.extraction),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "RunDiagnostics":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            layout=LayoutDiagnostics.from_dict(payload.get("layout")),
            ocr=dict(payload.get("ocr", {})) if isinstance(payload.get("ocr"), dict) else {},
            extraction=(
                dict(payload.get("extraction", {}))
                if isinstance(payload.get("extraction"), dict)
                else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: RunId
    input: RunInput
    pages: list[PageRef]
    created_at: str
    updated_at: str
    status: RunStatus = field(default_factory=RunStatus)
    diagnostics: RunDiagnostics = field(default_factory=RunDiagnostics)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": str(self.run_id),
            "input": self.input.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pages": [page.to_dict() for page in self.pages],
            "status": self.status.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, run_dir: Path | None = None) -> "RunManifest":
        return cls(
            schema_version=int(payload.get("schema_version", 0)),
            run_id=RunId(str(payload.get("run_id", ""))),
            input=RunInput.from_dict(payload.get("input", {})),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            pages=[
                PageRef.from_dict(page, run_dir=run_dir)
                for page in payload.get("pages", [])
                if isinstance(page, dict)
            ],
            status=RunStatus.from_dict(payload.get("status")),
            diagnostics=RunDiagnostics.from_dict(payload.get("diagnostics")),
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


@dataclass(frozen=True, slots=True)
class LayoutBlock:
    id: str
    index: int
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "label": self.label,
            "content": self.content,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LayoutBlock":
        raw_bbox = payload.get("bbox", [0, 0, 0, 0])
        bbox_values = raw_bbox if isinstance(raw_bbox, list) and len(raw_bbox) == 4 else [0, 0, 0, 0]
        return cls(
            id=str(payload.get("id", "")),
            index=int(payload.get("index", 0)),
            label=str(payload.get("label", "unknown")),
            content=str(payload.get("content", "")),
            confidence=float(payload.get("confidence", 1.0)),
            bbox=tuple(float(value) for value in bbox_values),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ReviewPage:
    page_number: int
    image_path: str
    image_width: int
    image_height: int
    blocks: list[LayoutBlock]
    provider_path: str | None = None
    coord_space: str = "pixel"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "page_number": self.page_number,
            "image_path": self.image_path,
            "image_size": {"width": self.image_width, "height": self.image_height},
            "coord_space": self.coord_space,
            "blocks": [block.to_dict() for block in self.blocks],
        }
        if self.provider_path:
            payload["provider_path"] = self.provider_path
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReviewPage":
        size = payload.get("image_size", {})
        return cls(
            page_number=int(payload.get("page_number", 0)),
            image_path=str(payload.get("image_path", payload.get("page_path", ""))),
            image_width=int(size.get("width", 0)) if isinstance(size, dict) else 0,
            image_height=int(size.get("height", 0)) if isinstance(size, dict) else 0,
            coord_space=str(payload.get("coord_space", "pixel")),
            provider_path=(
                str(payload.get("provider_path"))
                if payload.get("provider_path") is not None
                else None
            ),
            blocks=[
                LayoutBlock.from_dict(block)
                for block in payload.get("blocks", [])
                if isinstance(block, dict)
            ],
        )


@dataclass(frozen=True, slots=True)
class ReviewLayout:
    pages: list[ReviewPage]
    status: str = "prepared"
    version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "pages": [page.to_dict() for page in self.pages],
            "summary": {"page_count": len(self.pages)},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReviewLayout":
        return cls(
            version=int(payload.get("version", 2)),
            status=str(payload.get("status", "prepared")),
            pages=[
                ReviewPage.from_dict(page)
                for page in payload.get("pages", [])
                if isinstance(page, dict)
            ],
        )


@dataclass(frozen=True, slots=True)
class OcrPageResult:
    page_number: int
    image_path: str
    markdown: str
    markdown_source: str = "unknown"
    provider_path: str | None = None
    fallback_path: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "page_number": self.page_number,
            "image_path": self.image_path,
            "markdown": self.markdown,
            "markdown_source": self.markdown_source,
            "raw_payload": dict(self.raw_payload),
        }
        if self.provider_path:
            payload["provider_path"] = self.provider_path
        if self.fallback_path:
            payload["fallback_path"] = self.fallback_path
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OcrPageResult":
        return cls(
            page_number=int(payload.get("page_number", 0)),
            image_path=str(payload.get("image_path", payload.get("page_path", ""))),
            markdown=str(payload.get("markdown", "")),
            markdown_source=str(payload.get("markdown_source", "unknown")),
            provider_path=(
                str(payload.get("provider_path"))
                if payload.get("provider_path") is not None
                else None
            ),
            fallback_path=(
                str(payload.get("fallback_path"))
                if payload.get("fallback_path") is not None
                else None
            ),
            raw_payload=(
                dict(payload.get("raw_payload", {}))
                if isinstance(payload.get("raw_payload"), dict)
                else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class OcrRunResult:
    pages: list[OcrPageResult]
    markdown: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": [page.to_dict() for page in self.pages],
            "summary": {
                "page_count": len(self.pages),
                "sources": self.source_counts(),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, markdown: str = "") -> "OcrRunResult":
        pages = [
            OcrPageResult.from_dict(page)
            for page in payload.get("pages", [])
            if isinstance(page, dict)
        ]
        return cls(pages=pages, markdown=markdown or "\n\n".join(page.markdown for page in pages))

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for page in self.pages:
            counts[page.markdown_source] = counts.get(page.markdown_source, 0) + 1
        return counts


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


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    run_dir: Path
    manifest: RunManifest
    review_layout: ReviewLayout | None = None
    ocr_result: OcrRunResult | None = None
    extraction: dict[str, Any] = field(default_factory=dict)

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


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    snapshot: RunSnapshot
    warning: str | None = None
