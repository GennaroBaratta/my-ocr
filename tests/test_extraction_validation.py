from __future__ import annotations

from my_ocr.extraction.validation import validate_structured_prediction


def test_validate_structured_prediction_rejects_placeholder_scalars() -> None:
    validation = validate_structured_prediction(
        {
            "document_type": "unknown",
            "title": "unknown",
            "institution": "unknown",
            "date": "unknown",
            "language": "unknown",
            "summary_line": "unknown",
            "authors": [],
        }
    )

    assert validation["ok"] is False
    assert "placeholder value 'unknown'" in validation["reasons"][0]


def test_validate_structured_prediction_checks_values_against_source_text() -> None:
    validation = validate_structured_prediction(
        {
            "document_type": "report",
            "title": "Demo",
            "authors": ["Ada Lovelace"],
            "institution": "Missing University",
            "date": "2026-04-25",
            "language": "en",
            "summary_line": "OCR summary",
        },
        source_text="Report by Ada Lovelace for Example University.",
    )

    assert validation["ok"] is False
    assert "institution value 'Missing University' not found in OCR text" in validation["reasons"]
    assert "date value '2026-04-25' not found in OCR text" in validation["reasons"]


def test_validate_structured_prediction_rejects_schema_echo() -> None:
    validation = validate_structured_prediction(
        {
            "document_type": "report",
            "title": "Required: document_type, title, authors, institution, date, language",
            "authors": [],
            "institution": "",
            "date": "",
            "language": "en",
            "summary_line": "Required: return valid JSON only",
        }
    )

    assert validation["ok"] is False
    assert "title appears to echo extraction schema or instructions" in validation["reasons"]
    assert "summary_line appears to echo extraction schema or instructions" in validation["reasons"]
