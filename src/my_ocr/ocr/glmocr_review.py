from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from my_ocr.domain import LayoutBlock, PageRef, ReviewPage
from my_ocr.ocr.planning import extract_layout_blocks, normalize_bbox


def get_image_size(page_path: str | Path) -> tuple[int, int]:
    try:
        image_module = import_module("PIL.Image")
    except ImportError as exc:
        raise RuntimeError("Review preparation requires Pillow.") from exc

    with image_module.open(page_path) as image:
        return image.size


def review_layout_payload_from_page(
    review_page: ReviewPage | None,
) -> tuple[dict[str, Any], str] | None:
    if review_page is None:
        return None
    blocks = [
        {
            "index": block.index,
            "label": block.label,
            "content": block.content,
            "bbox_2d": block.bbox,
        }
        for block in review_page.blocks
    ]
    return {"blocks": blocks}, review_page.coord_space


def review_page_from_provider_layout(
    *,
    page_ref: PageRef,
    layout: Any,
    coord_space: str,
    image_width: int,
    image_height: int,
) -> ReviewPage:
    return ReviewPage(
        page_number=page_ref.page_number,
        image_path=page_ref.image_path,
        image_width=image_width,
        image_height=image_height,
        coord_space="pixel",
        provider_path=f"layout/provider/page-{page_ref.page_number:04d}",
        blocks=review_blocks_from_provider_layout(
            page_number=page_ref.page_number,
            layout=layout,
            coord_space=coord_space,
            image_width=image_width,
            image_height=image_height,
        ),
    )


def review_blocks_from_provider_layout(
    *,
    page_number: int,
    layout: Any,
    coord_space: str,
    image_width: int,
    image_height: int,
) -> list[LayoutBlock]:
    blocks: list[LayoutBlock] = []
    for fallback_index, block in enumerate(extract_layout_blocks(layout)):
        bbox = normalize_bbox(
            block.get("bbox_2d"),
            image_width,
            image_height,
            coord_space=coord_space,
        )
        if bbox is None:
            continue
        block_index = _coerce_block_index(block.get("index"), fallback=fallback_index)
        blocks.append(
            LayoutBlock(
                id=f"p{page_number - 1}-b{block_index}",
                index=block_index,
                label=str(block.get("label", "unknown")),
                content=str(block.get("content", "")),
                confidence=_coerce_confidence(block.get("confidence")),
                bbox=bbox,
            )
        )
    return blocks


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
