from __future__ import annotations

from typing import Any


def extract_layout_blocks(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "bbox_2d" in payload and "label" in payload:
            return [payload]
        blocks: list[dict[str, Any]] = []
        for value in payload.values():
            blocks.extend(extract_layout_blocks(value))
        return blocks
    if isinstance(payload, list):
        blocks: list[dict[str, Any]] = []
        for item in payload:
            blocks.extend(extract_layout_blocks(item))
        return blocks
    return []
