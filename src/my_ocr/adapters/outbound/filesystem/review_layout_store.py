from __future__ import annotations

from pathlib import Path
from typing import Any

from my_ocr.filesystem import read_json, write_json


def load_review_layout_payload(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    payload = read_json(candidate)
    if not isinstance(payload, dict):
        return None
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return None
    return payload


def save_review_layout_payload(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)

