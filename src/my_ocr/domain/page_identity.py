from __future__ import annotations

import re
from pathlib import Path


def infer_page_number(page_path: str | Path, fallback_number: int) -> int:
    match = re.fullmatch(r"page-(\d+)", Path(page_path).stem)
    if match is None:
        return fallback_number
    return int(match.group(1))

