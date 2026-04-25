from __future__ import annotations

from typing import Any

from my_ocr.extraction.validation import validate_structured_prediction


def choose_canonical_prediction(
    *,
    structured_prediction: dict[str, Any],
    rules_prediction: dict[str, Any] | None,
    source_text: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose canonical extraction output and emit metadata."""
    validation = validate_structured_prediction(structured_prediction, source_text=source_text)
    canonical = structured_prediction
    canonical_source = "structured"
    if not validation["ok"] and isinstance(rules_prediction, dict):
        canonical = rules_prediction
        canonical_source = "rules"
    metadata = {
        "canonical_source": canonical_source,
        "validation": validation,
    }
    return canonical, metadata
