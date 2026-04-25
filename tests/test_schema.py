import pytest
from pydantic import ValidationError

from my_ocr.domain.document import DocumentFields, JSON_SCHEMA
from my_ocr.models import LayoutBlock, PageRef, RunId, RunManifest, SCHEMA_VERSION


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


def test_persisted_models_reject_unknown_fields_and_wrong_types() -> None:
    with pytest.raises(ValidationError):
        PageRef.model_validate(
            {
                "page_number": "1",
                "image_path": "pages/page-0001.png",
                "width": 10,
                "height": 10,
            }
        )

    with pytest.raises(ValidationError):
        LayoutBlock.model_validate(
            {
                "id": "b1",
                "index": 0,
                "label": "text",
                "bbox": [1, 2, 3, 4],
                "unexpected": True,
            }
        )


def test_run_manifest_uses_schema_v3() -> None:
    manifest = RunManifest.new(RunId("demo"), "input.pdf")

    assert SCHEMA_VERSION == 3
    assert manifest.model_dump(mode="json")["schema_version"] == 3
