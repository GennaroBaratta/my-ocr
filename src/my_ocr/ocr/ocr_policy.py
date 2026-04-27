from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .text_cleanup import (
    clean_recognized_text,
    has_meaningful_text,
    reconstruct_markdown_from_layout,
)
from .labels import OCR_LABELS, TEXT_LABELS
from .layout_blocks import extract_layout_blocks


@dataclass(frozen=True, slots=True)
class PageOcrPlan:
    assessment: dict[str, Any]
    coord_space: str
    layout_markdown: str
    primary_source: Literal[
        "sdk_markdown",
        "layout_json",
        "crop_fallback",
        "full_page_fallback",
    ]
    fallback_source: Literal["full_page_fallback"] | None = None


def plan_page_ocr(
    sdk_markdown: str,
    page_json: Any,
    *,
    coord_space: str | None = None,
) -> PageOcrPlan:
    blocks = extract_layout_blocks(page_json)
    ocr_blocks = [block for block in blocks if block.get("label") in OCR_LABELS]
    text_blocks = [block for block in ocr_blocks if block.get("label") in TEXT_LABELS]
    meaningful_block_text = [
        clean_recognized_text(str(block.get("content", ""))) for block in ocr_blocks
    ]
    structured_layout_payload = isinstance(page_json, (dict, list))
    layout_markdown = reconstruct_markdown_from_layout(page_json)
    resolved_coord_space = coord_space or "unknown"

    if has_meaningful_text(sdk_markdown):
        primary_source = "sdk_markdown"
        reason = "markdown_present"
        fallback_source = None
    elif not structured_layout_payload:
        primary_source = "full_page_fallback"
        reason = "empty_markdown_and_unusable_json"
        fallback_source = None
    elif has_meaningful_text(layout_markdown):
        primary_source = "layout_json"
        reason = "empty_markdown_but_layout_has_text"
        fallback_source = None
    elif ocr_blocks:
        primary_source = "crop_fallback"
        reason = "empty_markdown_and_empty_layout_text"
        fallback_source = "full_page_fallback"
    elif blocks:
        primary_source = "full_page_fallback"
        reason = "empty_markdown_and_no_text_regions"
        fallback_source = None
    else:
        primary_source = "full_page_fallback"
        reason = "empty_markdown_and_no_layout_blocks"
        fallback_source = None

    assessment = {
        "use_fallback": primary_source in {"crop_fallback", "full_page_fallback"},
        "reason": reason,
        "structured_layout_payload": structured_layout_payload,
        "layout_block_count": len(blocks),
        "ocr_block_count": len(ocr_blocks),
        "text_block_count": len(text_blocks),
        "meaningful_text_block_count": sum(1 for text in meaningful_block_text if text),
        "bbox_coord_space": resolved_coord_space,
    }

    if primary_source == "sdk_markdown":
        return PageOcrPlan(
            assessment=assessment,
            coord_space=resolved_coord_space,
            layout_markdown=layout_markdown,
            primary_source=primary_source,
        )
    if primary_source == "layout_json":
        return PageOcrPlan(
            assessment=assessment,
            coord_space=resolved_coord_space,
            layout_markdown=layout_markdown,
            primary_source=primary_source,
        )
    if primary_source == "crop_fallback":
        return PageOcrPlan(
            assessment=assessment,
            coord_space=resolved_coord_space,
            layout_markdown=layout_markdown,
            primary_source=primary_source,
            fallback_source=fallback_source,
        )
    return PageOcrPlan(
        assessment=assessment,
        coord_space=resolved_coord_space,
        layout_markdown=layout_markdown,
        primary_source=primary_source,
    )
