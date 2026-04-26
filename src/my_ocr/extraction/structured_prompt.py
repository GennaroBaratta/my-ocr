from __future__ import annotations

import re

from my_ocr.support.text import replace_html_tables


def build_structured_prompt() -> str:
    return (
        "Extract only values explicitly present in the OCR text or image. "
        "Do not infer, guess, normalize, or repair missing content. "
        "Return valid JSON only. "
        "Leave any field that is not explicitly present empty (empty string or empty list). "
        "If unsure, use an empty value."
    )


def clean_structured_input_text(markdown_text: str) -> str:
    text = replace_html_tables(markdown_text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_meaningful_markdown(markdown_text: str | None) -> bool:
    if markdown_text is None:
        return False
    return any(char.isalnum() for char in markdown_text)


__all__ = [
    "build_structured_prompt",
    "clean_structured_input_text",
    "has_meaningful_markdown",
]
