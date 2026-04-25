from __future__ import annotations

from dataclasses import dataclass

from my_ocr.application.models import RunSnapshot


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    snapshot: RunSnapshot
    warning: str | None = None
