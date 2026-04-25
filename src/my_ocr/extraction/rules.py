from __future__ import annotations

import re
from typing import Any

from my_ocr.domain.document import DocumentFields
from my_ocr.support.text import collapse_whitespace

DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
    re.compile(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b"),
    re.compile(r"\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
]

INSTITUTION_KEYWORDS = (
    "university",
    "institute",
    "department",
    "college",
    "hospital",
    "center",
    "centre",
    "school",
    "laboratory",
    "ministry",
    "agency",
)

DOCUMENT_TYPE_MARKERS = (
    ("invoice", ("invoice",)),
    ("letter", ("dear ", "sincerely")),
    ("form", ("form", "application")),
    ("article", ("abstract", "references")),
    ("report", ("report",)),
)


def extract_from_markdown(markdown: str) -> dict[str, Any]:
    lines = _clean_lines(markdown)
    record = DocumentFields.empty()

    if not lines:
        return record.to_dict()

    record.title = _guess_title(lines)
    record.authors = _guess_authors(lines, record.title)
    record.institution = _guess_institution(lines)
    record.date = _guess_date(lines)
    record.language = _guess_language(markdown)
    record.document_type = _guess_document_type(markdown)
    record.summary_line = _guess_summary_line(lines, record.title, record.authors)
    return record.to_dict()


def _clean_lines(markdown: str) -> list[str]:
    cleaned = [
        collapse_whitespace(line.replace("#", " ").replace("*", " "))
        for line in markdown.splitlines()
    ]
    return [line for line in cleaned if len(line) >= 3]


def _guess_title(lines: list[str]) -> str:
    title_candidates = [
        line for line in lines[:6] if not re.search(r"\b(?:page|doi|abstract)\b", line, re.I)
    ]
    if not title_candidates:
        return ""
    first = title_candidates[0]
    if len(title_candidates) > 1 and _should_merge_title_lines(first, title_candidates[1]):
        combined = collapse_whitespace(f"{first} {title_candidates[1]}")
        return combined if len(combined) <= 160 else first
    return first


def _guess_authors(lines: list[str], title: str) -> list[str]:
    title_index = lines.index(title) if title in lines else 0
    search_space = lines[title_index + 1 : title_index + 4]
    for line in search_space:
        if re.search(r"\d", line):
            continue
        if any(keyword in line.lower() for keyword in INSTITUTION_KEYWORDS):
            continue
        if len(line.split()) > 12:
            continue
        chunks = re.split(r",|;|\band\b", line, flags=re.I)
        authors = [collapse_whitespace(chunk) for chunk in chunks if 1 <= len(chunk.split()) <= 5]
        if authors:
            return authors
    return []


def _looks_like_author_line(line: str) -> bool:
    if "," in line or ";" in line:
        return True
    words = line.split()
    alpha_words = [word for word in words if word.isalpha()]
    if 2 <= len(alpha_words) <= 8 and all(word[:1].isupper() for word in alpha_words):
        return True
    return False


def _guess_institution(lines: list[str]) -> str:
    for line in lines[:12]:
        lowered = line.lower()
        if any(keyword in lowered for keyword in INSTITUTION_KEYWORDS):
            return line
    return ""


def _guess_date(lines: list[str]) -> str:
    text = "\n".join(lines[:20])
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return ""


def _guess_language(markdown: str) -> str:
    lowered = markdown.lower()
    english_markers = {
        "the",
        "and",
        "of",
        "for",
        "this",
        "report",
        "document",
        "summary",
        "university",
    }
    tokens = set(re.findall(r"[a-z]+", lowered))
    if len(tokens & english_markers) >= 2:
        return "en"
    return ""


def _guess_document_type(markdown: str) -> str:
    lowered = markdown.lower()
    for document_type, markers in DOCUMENT_TYPE_MARKERS:
        if any(marker in lowered for marker in markers):
            return document_type
    return "document"


def _guess_summary_line(lines: list[str], title: str, authors: list[str]) -> str:
    skip = {title, *authors}
    for line in lines[1:10]:
        if line in skip:
            continue
        if 8 <= len(line.split()) <= 30:
            return line
    return ""


def _should_merge_title_lines(first: str, second: str) -> bool:
    if len(first.split()) >= 5 or len(second.split()) >= 12:
        return False
    if _looks_like_author_line(second):
        return False
    return not any(keyword in second.lower() for keyword in INSTITUTION_KEYWORDS)
