from __future__ import annotations

import json
from typing import Any, cast

from .state import AppState


def current_page_index(state: AppState, markdown_pages: list[str]) -> int:
    if not markdown_pages:
        return 0
    return min(max(state.session.current_page_index, 0), len(markdown_pages) - 1)


def markdown_pages_for_state(state: AppState) -> list[str]:
    pages = ocr_pages_for_state(state)
    if pages:
        markdown_pages = [
            page.get("markdown", "") if isinstance(page.get("markdown"), str) else ""
            for page in pages
        ]
        target_count = max(len(markdown_pages), len(state.session.pages), 1)
        while len(markdown_pages) < target_count:
            markdown_pages.append("")
        return markdown_pages
    if state.session.ocr_markdown.strip():
        return [state.session.ocr_markdown]
    return [""]


def raw_page_text_for_state(state: AppState) -> str:
    markdown_pages = markdown_pages_for_state(state)
    page_index = current_page_index(state, markdown_pages)
    pages = ocr_pages_for_state(state)
    if 0 <= page_index < len(pages):
        return json.dumps(pages[page_index], indent=2, ensure_ascii=False)
    return "No OCR page payload available for this page."


def current_page_markdown_for_state(state: AppState) -> str:
    markdown_pages = markdown_pages_for_state(state)
    page_index = current_page_index(state, markdown_pages)
    if 0 <= page_index < len(markdown_pages):
        return markdown_pages[page_index]
    return ""


def current_page_ocr_markdown_for_state(state: AppState) -> str:
    pages = ocr_pages_for_state(state)
    if not pages:
        return ""
    page_index = state.session.current_page_index
    if not 0 <= page_index < len(pages):
        return ""
    markdown = pages[page_index].get("markdown")
    return markdown if isinstance(markdown, str) else ""


def ocr_json_text_for_state(state: AppState) -> str:
    return json.dumps(state.session.ocr_json, indent=2, ensure_ascii=False) if state.session.ocr_json else ""


def ocr_pages_for_state(state: AppState) -> list[dict[str, Any]]:
    payload = state.session.ocr_json
    if not isinstance(payload, dict):
        return []
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return []
    return [cast(dict[str, Any], page) for page in pages if isinstance(page, dict)]

