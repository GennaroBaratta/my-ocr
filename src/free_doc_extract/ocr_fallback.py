from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .ollama_client import encode_image_file, post_json
from .settings import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_MODEL,
)
from .utils import ensure_dir, write_text

TEXT_RECOGNITION_PROMPT = "Text Recognition: [img-0]"
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
BOX_PADDING_X = 8
BOX_PADDING_Y = 6


def needs_crop_fallback(markdown: str, page_json: Any) -> bool:
    blocks = extract_layout_blocks(page_json)
    if not blocks:
        return False

    meaningful_block_text = [
        clean_recognized_text(str(block.get("content", "")))
        for block in blocks
        if block.get("label") in TEXT_LABELS
    ]
    if any(meaningful_block_text):
        return False

    return not has_meaningful_text(markdown)


def run_crop_fallback_for_page(
    *,
    page_path: str,
    page_json: Any,
    page_fallback_dir: str | Path,
    model: str = DEFAULT_OLLAMA_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
) -> tuple[str, list[dict[str, Any]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Crop-based OCR fallback requires Pillow.") from exc

    page_fallback_dir = ensure_dir(page_fallback_dir)
    with Image.open(page_path) as image:
        width, height = image.size
        chunks = build_text_chunks(page_json, width=width, height=height)

        recognized_chunks: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            crop_path = page_fallback_dir / f"chunk-{chunk_index:04d}.png"
            text_path = page_fallback_dir / f"chunk-{chunk_index:04d}.txt"
            cropped = image.crop(tuple(chunk["bbox"]))
            cropped.save(crop_path)

            try:
                recognized_text = recognize_text_crop(crop_path, model=model, endpoint=endpoint)
            except RuntimeError as exc:
                recognized_text = ""
                error = str(exc)
            else:
                error = ""

            write_text(text_path, recognized_text)
            recognized_chunks.append(
                {
                    "chunk": chunk_index,
                    "bbox": chunk["bbox"],
                    "labels": chunk["labels"],
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
) -> str:
    body = post_json(
        endpoint=endpoint,
        payload={
            "model": model,
            "prompt": TEXT_RECOGNITION_PROMPT,
            "images": [encode_image_file(crop_path)],
            "stream": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
        },
        error_prefix="Ollama crop OCR",
    )
    return clean_recognized_text(str(body.get("response", "")))


def build_text_chunks(page_json: Any, *, width: int, height: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for block in extract_layout_blocks(page_json):
        bbox = normalize_bbox(block.get("bbox_2d"), width, height)
        if block.get("label") in TEXT_LABELS and bbox is not None:
            chunks.append(
                {
                    "bbox": pad_bbox(bbox, width, height),
                    "labels": [str(block.get("label", ""))],
                    "source_indices": [safe_int(block.get("index"), -1)],
                    "unpadded_bbox": bbox,
                }
            )

    chunks.sort(key=lambda chunk: (chunk["unpadded_bbox"][1], chunk["unpadded_bbox"][0]))
    return chunks


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


def normalize_bbox(raw_bbox: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return None

    if max(x1, y1, x2, y2) <= 1000:
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
    if not has_meaningful_text(text):
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line.strip()).strip()


def has_meaningful_text(value: str) -> bool:
    return any(char.isalnum() for char in value)


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
