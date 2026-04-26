from __future__ import annotations

import re

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
