from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = 3


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class StrictModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )


def validate_optional_run_relative_path(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return validate_run_relative_path(value, field_name=field_name)


def validate_run_relative_path(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    path = Path(value)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or ".." in path.parts
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"{field_name} must be run-relative")
    return value
