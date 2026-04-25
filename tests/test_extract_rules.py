from my_ocr.pipeline.extraction import extract_from_markdown


def test_extract_from_markdown_returns_expected_baseline_fields() -> None:
    markdown = """
    # Sample Report
    Ada Lovelace, Grace Hopper
    Example University
    January 15, 2024
    This report describes a compact OCR evaluation pipeline for local document parsing.
    """

    result = extract_from_markdown(markdown)

    assert result["title"] == "Sample Report"
    assert result["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert result["institution"] == "Example University"
    assert result["date"] == "January 15, 2024"
    assert result["document_type"] == "report"
    assert result["language"] == "en"
    assert "compact OCR evaluation pipeline" in result["summary_line"]
