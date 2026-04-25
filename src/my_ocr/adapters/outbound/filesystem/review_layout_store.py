from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_review_layout_payload(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return None
    return payload


def save_review_layout_payload(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

