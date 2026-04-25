from __future__ import annotations

import re
from datetime import UTC, datetime



def timestamp_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "run"


def collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
