from my_ocr.schema import DocumentFields, JSON_SCHEMA


def test_document_fields_normalizes_mapping() -> None:
    payload = {
        "document_type": " report ",
        "title": "  Sample   Title  ",
        "authors": [" Ada Lovelace ", "", "Grace Hopper"],
        "institution": " Example University ",
        "date": " 2024-02-01 ",
        "language": " en ",
        "summary_line": " Short   summary. ",
    }

    record = DocumentFields.from_mapping(payload)

    assert record.document_type == "report"
    assert record.title == "Sample Title"
    assert record.authors == ["Ada Lovelace", "Grace Hopper"]
    assert record.institution == "Example University"
    assert record.date == "2024-02-01"
    assert record.language == "en"
    assert record.summary_line == "Short summary."


def test_json_schema_contains_required_fields() -> None:
    assert JSON_SCHEMA["type"] == "object"
    assert set(JSON_SCHEMA["required"]) == {
        "document_type",
        "title",
        "authors",
        "institution",
        "date",
        "language",
        "summary_line",
    }
