from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from .text import normalize_table_html

TEXT_RECOGNITION_PROMPT = "Text Recognition:"
TABLE_RECOGNITION_PROMPT = "Table Recognition:"
FORMULA_RECOGNITION_PROMPT = "Formula Recognition:"
TEXT_LABELS = {
    "abstract",
    "algorithm",
    "content",
    "doc_title",
    "figure_title",
    "paragraph_title",
    "reference_content",
    "text",
    "vertical_text",
    "vision_footnote",
    "seal",
    "formula_number",
}
TABLE_LABELS = {"table"}
FORMULA_LABELS = {"formula", "display_formula", "inline_formula"}
OCR_LABELS = TEXT_LABELS | TABLE_LABELS | FORMULA_LABELS
SYMBOLIC_CONTENT_RE = re.compile(r"[=+*/\\^_()\[\]{}<>∑∫√≈≠≤≥×÷]")
BOX_PADDING_X = 8
BOX_PADDING_Y = 6


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


def resolve_ocr_task(label: str) -> str:
    if label in TABLE_LABELS:
        return "table"
    if label in FORMULA_LABELS:
        return "formula"
    return "text"


def resolve_prompt_for_label(label: str) -> str:
    task = resolve_ocr_task(label)
    if task == "table":
        return TABLE_RECOGNITION_PROMPT
    if task == "formula":
        return FORMULA_RECOGNITION_PROMPT
    return TEXT_RECOGNITION_PROMPT


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


def _sort_key_for_block(raw_bbox: Any) -> tuple[float, float, float, float]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    try:
        x1, y1, x2, y2 = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    return (y1, x1, y2, x2)


def normalize_bbox(
    raw_bbox: Any,
    width: int,
    height: int,
    *,
    coord_space: str | None = None,
) -> list[int] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return None

    if coord_space == "normalized":
        x1 = round(x1 * width / 1000)
        x2 = round(x2 * width / 1000)
        y1 = round(y1 * height / 1000)
        y2 = round(y2 * height / 1000)
    elif coord_space == "normalized_unit":
        x1 = round(x1 * width)
        x2 = round(x2 * width)
        y1 = round(y1 * height)
        y2 = round(y2 * height)
    elif coord_space == "pixel":
        x1, y1, x2, y2 = [round(value) for value in (x1, y1, x2, y2)]
    else:
        if max(x1, y1, x2, y2) <= 1:
            x1 = round(x1 * width)
            x2 = round(x2 * width)
            y1 = round(y1 * height)
            y2 = round(y2 * height)
        elif max(x1, y1, x2, y2) <= 1000 and (width > 1000 or height > 1000):
            x1 = round(x1 * width / 1000)
            x2 = round(x2 * width / 1000)
            y1 = round(y1 * height / 1000)
            y2 = round(y2 * height / 1000)
        else:
            x1, y1, x2, y2 = [round(value) for value in (x1, y1, x2, y2)]

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def pad_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    return [
        max(0, bbox[0] - BOX_PADDING_X),
        max(0, bbox[1] - BOX_PADDING_Y),
        min(width, bbox[2] + BOX_PADDING_X),
        min(height, bbox[3] + BOX_PADDING_Y),
    ]


def clean_recognized_text(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    if "<table" in text.lower() and "</table>" in text.lower():
        text = normalize_table_html(text)
    if not has_meaningful_text(text):
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line.strip()).strip()


def has_meaningful_text(value: str) -> bool:
    if any(char.isalnum() for char in value):
        return True
    return bool(SYMBOLIC_CONTENT_RE.search(value))


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def detect_bbox_coord_space(
    blocks: list[dict[str, Any]], *, width: int | None = None, height: int | None = None
) -> str:
    coords: list[float] = []
    per_axis: list[tuple[float, float, float, float]] = []
    for block in blocks:
        bbox = block.get("bbox_2d")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox]
            except (TypeError, ValueError):
                continue
            coords.extend((x1, y1, x2, y2))
            per_axis.append((x1, y1, x2, y2))
    if not coords:
        return "unknown"
    if all(0 <= value <= 1 for value in coords):
        return "normalized_unit"
    if any(value > 1000 for value in coords):
        return "pixel"
    if all(0 <= value <= 1000 for value in coords):
        return "normalized"
    if width is None or height is None:
        return "unknown"

    fits_pixel_bounds = all(
        0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height for x1, y1, x2, y2 in per_axis
    )
    exceeds_pixel_bounds = any(x2 > width or y2 > height for _, _, x2, y2 in per_axis)

    if exceeds_pixel_bounds and all(0 <= value <= 1000 for value in coords):
        return "normalized"
    if fits_pixel_bounds:
        return "pixel"
    return "unknown"
