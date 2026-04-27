from __future__ import annotations

import re
from typing import Any

from .bbox import _sort_key_for_block
from .labels import OCR_LABELS
from .layout_blocks import extract_layout_blocks
from my_ocr.support.text import normalize_table_html

SYMBOLIC_CONTENT_RE = re.compile(r"[=+*/\\^_()\[\]{}<>∑∫√≈≠≤≥×÷]")


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


__all__ = [
    "clean_recognized_text",
    "has_meaningful_text",
    "normalize_table_html",
    "reconstruct_markdown_from_layout",
]
