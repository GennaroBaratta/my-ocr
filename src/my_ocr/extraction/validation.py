from __future__ import annotations

import re
from typing import Any

from my_ocr.support.text import collapse_whitespace


def validate_structured_prediction(
    prediction: dict[str, Any], *, source_text: str | None = None
) -> dict[str, Any]:
    reasons: list[str] = []
    scalar_field_names = (
        "document_type",
        "title",
        "institution",
        "date",
        "language",
        "summary_line",
    )
    normalized_scalars = [
        collapse_whitespace(str(prediction.get(name, ""))).lower() for name in scalar_field_names
    ]
    repeated_placeholder = {value for value in normalized_scalars if value}
    if len(repeated_placeholder) == 1:
        placeholder = next(iter(repeated_placeholder))
        if placeholder in {"document", "unknown", "n/a", "none", "null"}:
            reasons.append(f"all scalar fields collapsed to placeholder value {placeholder!r}")

    authors = prediction.get("authors") or []
    if not isinstance(authors, list):
        authors = [str(authors)]
    suspicious_author_tokens = {"[", "]", "{", "}", "[{", "}]", "```", "json"}
    if any(collapse_whitespace(str(author)) in suspicious_author_tokens for author in authors):
        reasons.append("authors field contains JSON fence or bracket fragments")

    summary_line = collapse_whitespace(str(prediction.get("summary_line", "")))
    if summary_line.lower().startswith("required:"):
        reasons.append("summary_line appears to echo extraction instructions")

    if source_text:
        normalized_source_text = _normalize_validation_text(source_text)
        for author in authors:
            author_text = collapse_whitespace(str(author))
            if author_text and not _text_present_in_source(author_text, normalized_source_text):
                reasons.append(f"author {author_text!r} not found in OCR text")

        for field_name in ("institution", "date"):
            field_value = collapse_whitespace(str(prediction.get(field_name, "")))
            if field_value and not _text_present_in_source(field_value, normalized_source_text):
                reasons.append(f"{field_name} value {field_value!r} not found in OCR text")

    return {"ok": not reasons, "reasons": reasons}


def _normalize_validation_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _text_present_in_source(value: str, normalized_source_text: str) -> bool:
    normalized_value = _normalize_validation_text(value)
    if not normalized_value:
        return True
    return normalized_value in normalized_source_text
