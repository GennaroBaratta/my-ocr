from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .utils import collapse_whitespace


JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "institution": {"type": "string"},
        "date": {"type": "string"},
        "language": {"type": "string"},
        "summary_line": {"type": "string"},
    },
    "required": [
        "document_type",
        "title",
        "authors",
        "institution",
        "date",
        "language",
        "summary_line",
    ],
}


@dataclass(slots=True)
class DocumentFields:
    document_type: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    institution: str = ""
    date: str = ""
    language: str = ""
    summary_line: str = ""

    @classmethod
    def empty(cls) -> "DocumentFields":
        return cls()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "DocumentFields":
        payload = payload or {}
        authors = payload.get("authors") or []
        if not isinstance(authors, list):
            authors = [str(authors)]

        return cls(
            document_type=_clean_string(payload.get("document_type", "")),
            title=_clean_string(payload.get("title", "")),
            authors=[_clean_string(author) for author in authors if _clean_string(str(author))],
            institution=_clean_string(payload.get("institution", "")),
            date=_clean_string(payload.get("date", "")),
            language=_clean_string(payload.get("language", "")),
            summary_line=_clean_string(payload.get("summary_line", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_string(value: Any) -> str:
    return collapse_whitespace(str(value)) if value is not None else ""
