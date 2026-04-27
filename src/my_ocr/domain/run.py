from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, RootModel, field_serializer, field_validator

from my_ocr.domain._base import SCHEMA_VERSION, StrictModel, utc_now_iso
from my_ocr.domain._base import validate_run_relative_path

InvalidatedArtifactGroup = Literal["ocr", "extraction"]


@dataclass(frozen=True, slots=True)
class RunInvalidationPlan:
    artifact_groups: tuple[InvalidatedArtifactGroup, ...]
    status: RunStatus | None = None
    diagnostics: RunDiagnostics | None = None

    @property
    def updates_manifest(self) -> bool:
        return self.status is not None or self.diagnostics is not None


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

    @field_validator("image_path")
    @classmethod
    def _validate_image_path(cls, value: str) -> str:
        return validate_run_relative_path(value, field_name="image_path")

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
    layout: Literal["pending", "prepared", "reviewed"] = "pending"
    ocr: Literal["pending", "complete"] = "pending"
    extraction: Literal["pending", "rules", "structured"] = "pending"


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
    schema_version: Literal[3] = SCHEMA_VERSION

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
