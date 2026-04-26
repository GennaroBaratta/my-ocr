from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .bbox import (
    BOX_PADDING_X as BOX_PADDING_X,
    BOX_PADDING_Y as BOX_PADDING_Y,
    _sort_key_for_block,
    detect_bbox_coord_space,
    normalize_bbox,
    pad_bbox,
)
from .labels import (
    FORMULA_LABELS as FORMULA_LABELS,
    FORMULA_RECOGNITION_PROMPT as FORMULA_RECOGNITION_PROMPT,
    OCR_LABELS,
    TABLE_LABELS as TABLE_LABELS,
    TABLE_RECOGNITION_PROMPT as TABLE_RECOGNITION_PROMPT,
    TEXT_LABELS,
    TEXT_RECOGNITION_PROMPT as TEXT_RECOGNITION_PROMPT,
    resolve_ocr_task,
    resolve_prompt_for_label,
)
from .layout_blocks import extract_layout_blocks
from .text_cleanup import clean_recognized_text, has_meaningful_text
from my_ocr.support.text import normalize_table_html as normalize_table_html


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


def reconstruct_markdown_from_layout(page_json: Any) -> str:
    ordered_blocks: list[tuple[tuple[float, float, float, float], str]] = []
    for block in extract_layout_blocks(page_json):
        if block.get("label") not in OCR_LABELS:
            continue
        text = clean_recognized_text(str(block.get("content", "")))
        if not text:
            continue
        ordered_blocks.append((_sort_key_for_block(block.get("bbox_2d")), text))

    # Layout-only recovery is a best-effort salvage path, so bbox ordering is kept simple.
    ordered_blocks.sort(key=lambda item: item[0])
    return "\n\n".join(text for _, text in ordered_blocks)


def build_ocr_chunks(
    page_json: Any,
    *,
    width: int,
    height: int,
    coord_space: str | None = None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    blocks = extract_layout_blocks(page_json)
    effective_coord_space = coord_space
    if effective_coord_space in (None, "unknown"):
        effective_coord_space = detect_bbox_coord_space(blocks, width=width, height=height)
    for block_offset, block in enumerate(blocks):
        label = str(block.get("label", ""))
        bbox = normalize_bbox(
            block.get("bbox_2d"), width, height, coord_space=effective_coord_space
        )
        if label in OCR_LABELS and bbox is not None:
            source_index = block.get("index")
            if source_index is None:
                source_index = block_offset
            chunks.append(
                {
                    "bbox": pad_bbox(bbox, width, height),
                    "labels": [label],
                    "task": resolve_ocr_task(label),
                    "prompt": resolve_prompt_for_label(label),
                    "source_indices": [safe_int(source_index, block_offset)],
                    "unpadded_bbox": bbox,
                }
            )

    chunks.sort(key=lambda chunk: (chunk["unpadded_bbox"][1], chunk["unpadded_bbox"][0]))
    return chunks


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
