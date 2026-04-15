from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .ollama_client import encode_image_file, post_json
from .settings import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
)
from .utils import ensure_dir, write_text

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
FORMULA_LABELS = {"display_formula", "inline_formula"}
OCR_LABELS = TEXT_LABELS | TABLE_LABELS | FORMULA_LABELS
SYMBOLIC_CONTENT_RE = re.compile(r"[=+*/\\^_()\[\]{}<>∑∫√≈≠≤≥×÷]")
BOX_PADDING_X = 8
BOX_PADDING_Y = 6


def assess_crop_fallback(
    markdown: str,
    page_json: Any,
    *,
    coord_space: str | None = None,
) -> dict[str, Any]:
    blocks = extract_layout_blocks(page_json)
    ocr_blocks = [block for block in blocks if block.get("label") in OCR_LABELS]
    text_blocks = [block for block in ocr_blocks if block.get("label") in TEXT_LABELS]
    meaningful_block_text = [
        clean_recognized_text(str(block.get("content", ""))) for block in ocr_blocks
    ]
    meaningful_markdown = has_meaningful_text(markdown)
    structured_layout_payload = isinstance(page_json, (dict, list))

    if meaningful_markdown:
        reason = "markdown_present"
        use_fallback = False
    elif not structured_layout_payload:
        reason = "empty_markdown_and_unusable_json"
        use_fallback = True
    elif any(meaningful_block_text):
        reason = "empty_markdown_but_layout_has_text"
        use_fallback = False
    elif ocr_blocks:
        reason = "empty_markdown_and_empty_layout_text"
        use_fallback = True
    elif blocks:
        reason = "empty_markdown_and_no_text_regions"
        use_fallback = True
    else:
        reason = "empty_markdown_and_no_layout_blocks"
        use_fallback = True

    return {
        "use_fallback": use_fallback,
        "reason": reason,
        "structured_layout_payload": structured_layout_payload,
        "layout_block_count": len(blocks),
        "ocr_block_count": len(ocr_blocks),
        "text_block_count": len(text_blocks),
        "meaningful_text_block_count": sum(1 for text in meaningful_block_text if text),
        "bbox_coord_space": coord_space or "unknown",
    }


def needs_crop_fallback(markdown: str, page_json: Any) -> bool:
    return bool(assess_crop_fallback(markdown, page_json)["use_fallback"])


def run_crop_fallback_for_page(
    *,
    page_path: str,
    page_json: Any,
    coord_space: str | None = None,
    page_fallback_dir: str | Path,
    model: str = DEFAULT_OLLAMA_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
) -> tuple[str, list[dict[str, Any]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Crop-based OCR fallback requires Pillow.") from exc

    page_fallback_dir = ensure_dir(page_fallback_dir)
    with Image.open(page_path) as image:
        width, height = image.size
        chunks = build_ocr_chunks(page_json, width=width, height=height, coord_space=coord_space)

        recognized_chunks: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            crop_path = page_fallback_dir / f"chunk-{chunk_index:04d}.png"
            text_path = page_fallback_dir / f"chunk-{chunk_index:04d}.txt"
            cropped = image.crop(tuple(chunk["bbox"]))
            cropped.save(crop_path)

            try:
                recognized_text = recognize_text_image(
                    crop_path,
                    prompt=chunk["prompt"],
                    model=model,
                    endpoint=endpoint,
                    num_ctx=num_ctx,
                )
            except RuntimeError as exc:
                recognized_text = ""
                error = str(exc)
            else:
                error = ""

            recognized_text = clean_recognized_text(recognized_text)
            write_text(text_path, recognized_text)
            recognized_chunks.append(
                {
                    "chunk": chunk_index,
                    "bbox": chunk["bbox"],
                    "labels": chunk["labels"],
                    "task": chunk["task"],
                    "prompt": chunk["prompt"],
                    "source_indices": chunk["source_indices"],
                    "crop_path": str(crop_path),
                    "text_path": str(text_path),
                    "text": recognized_text,
                    "error": error,
                }
            )

    page_markdown = "\n\n".join(
        chunk["text"].strip() for chunk in recognized_chunks if chunk["text"].strip()
    )
    return page_markdown, recognized_chunks


def recognize_text_crop(
    crop_path: str | Path,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
) -> str:
    return recognize_text_image(crop_path, model=model, endpoint=endpoint, num_ctx=num_ctx)


def recognize_full_page(
    page_path: str | Path,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
) -> str:
    return recognize_text_image(page_path, model=model, endpoint=endpoint, num_ctx=num_ctx)


def recognize_text_image(
    image_path: str | Path,
    *,
    prompt: str = TEXT_RECOGNITION_PROMPT,
    model: str = DEFAULT_OLLAMA_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
) -> str:
    body = post_json(
        endpoint=endpoint,
        payload={
            "model": model,
            "prompt": prompt,
            "images": [encode_image_file(image_path)],
            "stream": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {"num_ctx": num_ctx},
        },
        error_prefix="Ollama crop OCR",
    )
    return clean_recognized_text(str(body.get("response", "")))


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


def build_text_chunks(
    page_json: Any,
    *,
    width: int,
    height: int,
    coord_space: str | None = None,
) -> list[dict[str, Any]]:
    return build_ocr_chunks(page_json, width=width, height=height, coord_space=coord_space)


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


class _TableHtmlToTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs):
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._current_cell = []

    def handle_endtag(self, tag: str):
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            cell_text = " ".join("".join(self._current_cell).split())
            self._current_row.append(cell_text)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str):
        if self._current_cell is not None:
            self._current_cell.append(data)


def normalize_table_html(text: str) -> str:
    parser = _TableHtmlToTextParser()
    parser.feed(text)
    parser.close()
    rows = [row for row in parser.rows if any(cell.strip() for cell in row)]
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows).strip()


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
