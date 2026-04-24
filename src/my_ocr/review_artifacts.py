from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .ocr_fallback import extract_layout_blocks, normalize_bbox
from .utils import load_json, write_json

REVIEW_LAYOUT_VERSION = 1


def build_review_layout_payload(pages: Sequence[dict[str, Any]], *, status: str) -> dict[str, Any]:
    return {
        "version": REVIEW_LAYOUT_VERSION,
        "status": status,
        "pages": list(pages),
        "summary": {"page_count": len(pages)},
    }


def build_review_page_from_layout(
    *,
    page_number: int,
    page_path: str,
    source_sdk_json_path: str,
    layout: Any,
    coord_space: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for fallback_index, block in enumerate(extract_layout_blocks(layout)):
        bbox = normalize_bbox(
            block.get("bbox_2d"),
            image_width,
            image_height,
            coord_space=coord_space,
        )
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        block_index = _coerce_block_index(block.get("index"), fallback=fallback_index)
        blocks.append(
            {
                "id": f"p{page_number - 1}-b{block_index}",
                "index": block_index,
                "label": str(block.get("label", "unknown")),
                "content": str(block.get("content", "")),
                "confidence": _coerce_confidence(block.get("confidence")),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return {
        "page_number": page_number,
        "page_path": page_path,
        "image_size": {"width": image_width, "height": image_height},
        "source_sdk_json_path": source_sdk_json_path,
        "coord_space": "pixel",
        "blocks": blocks,
    }


def load_review_layout_payload(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    payload = load_json(candidate)
    if not isinstance(payload, dict):
        return None
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return None
    return payload


def save_review_layout_payload(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def review_layout_pages_by_number(payload: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if payload is None:
        return {}
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return {}

    indexed: dict[int, dict[str, Any]] = {}
    for fallback_index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        if isinstance(page_number, int) and page_number > 0:
            indexed[page_number] = page
        else:
            indexed[fallback_index + 1] = page
    return indexed


def review_page_to_layout_payload(page: dict[str, Any]) -> tuple[dict[str, Any], str]:
    blocks_payload = page.get("blocks")
    if not isinstance(blocks_payload, list):
        return {"blocks": []}, "pixel"

    blocks: list[dict[str, Any]] = []
    for fallback_index, block in enumerate(blocks_payload):
        if not isinstance(block, dict):
            continue
        bbox = block.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        blocks.append(
            {
                "index": _coerce_block_index(block.get("index"), fallback=fallback_index),
                "label": str(block.get("label", "unknown")),
                "content": str(block.get("content", "")),
                "bbox_2d": bbox,
            }
        )

    coord_space = page.get("coord_space")
    return {"blocks": blocks}, coord_space if isinstance(coord_space, str) else "pixel"


def _coerce_block_index(value: Any, *, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0
