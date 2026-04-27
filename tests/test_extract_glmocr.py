from __future__ import annotations

from typing import Any

import pytest

from my_ocr.extraction.structured import (
    _clean_structured_input_text,
    _parse_response_json,
    build_structured_markdown_request,
    build_structured_prompt,
    extract_structured,
)
from my_ocr.inference import InferenceRequest, InferenceResponse, ProviderRequestError


class FakeInferenceClient:
    def __init__(
        self,
        *,
        text: str = '{"document_type":"report","authors":[]}',
        raw: dict[str, Any] | None = None,
        model: str | None = "fake-model",
        error: Exception | None = None,
    ) -> None:
        self.requests: list[InferenceRequest] = []
        self._text = text
        self._raw = raw or {"provider": "fake"}
        self._model = model
        self._error = error

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return InferenceResponse(text=self._text, raw=self._raw, model=self._model)


def test_parse_response_json_handles_fenced_json() -> None:
    parsed = _parse_response_json('```json\n{"title": "Demo"}\n```')
    assert parsed == {"title": "Demo"}


def test_parse_response_json_rejects_non_object() -> None:
    with pytest.raises(RuntimeError, match="was not a JSON object"):
        _parse_response_json("[1, 2, 3]")


def test_extract_structured_parses_provider_neutral_response_and_metadata(tmp_path) -> None:
    image1 = tmp_path / "page-0001.png"
    image2 = tmp_path / "page-0002.png"
    image1.write_bytes(b"fake image 1")
    image2.write_bytes(b"fake image 2")
    client = FakeInferenceClient(
        text='```json\n{"document_type": "report", "title": "Sample", "authors": [], "institution": "", "date": "", "language": "en", "summary_line": "Hi"}\n```',
        raw={"provider": "fake", "request_id": "abc"},
    )

    parsed, metadata = extract_structured(
        [str(image1), str(image2)], inference_client=client
    )

    request = client.requests[0]
    assert len(request.images) == 1
    assert request.images[0].path == str(image1)
    assert request.structured_output is not None
    assert request.structured_output.schema is not None
    assert parsed["title"] == "Sample"
    assert metadata == {
        "model": "fake-model",
        "source": "page_images_first_page",
        "input_length": None,
        "image_paths": [str(image1)],
        "input_page_count": 2,
        "used_page_count": 1,
        "_raw_body": {"provider": "fake", "request_id": "abc"},
    }


def test_extract_structured_uses_markdown_text_when_meaningful(tmp_path) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")
    client = FakeInferenceClient(text='{"document_type":"report","authors":[]}')

    parsed, metadata = extract_structured(
        [str(image)], markdown_text="# OCR\nSome text", inference_client=client
    )

    request = client.requests[0]
    assert request.images == ()
    assert "OCR text:\n# OCR\nSome text" in request.prompt
    assert parsed["document_type"] == "report"
    assert metadata["source"] == "ocr_markdown"
    assert metadata["input_length"] == len("# OCR\nSome text")
    assert metadata["image_paths"] == []
    assert metadata["used_page_count"] == 0


def test_structured_prompt_emphasizes_explicit_values_only() -> None:
    prompt = build_structured_prompt()

    assert "explicitly present" in prompt
    assert "Do not infer" in prompt
    assert "JSON_SCHEMA" not in prompt


def test_clean_structured_input_text_preserves_table_content() -> None:
    cleaned = _clean_structured_input_text(
        "<p>Hello</p><table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>1</td></tr></table>"
    )

    assert "Hello" in cleaned
    assert "Name | Value" in cleaned
    assert "A | 1" in cleaned
    assert "<p>" not in cleaned


def test_structured_markdown_request_uses_cleaned_text_and_schema() -> None:
    request = build_structured_markdown_request(
        "<div>  A   B </div><table><tr><td>X</td><td>Y</td></tr></table>",
        model="demo",
    )

    assert request.model == "demo"
    assert request.prompt.endswith("OCR text:\nA B X | Y")
    assert request.structured_output is not None
    assert request.structured_output.schema is not None


def test_extract_structured_ignores_separator_only_markdown(tmp_path) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")
    client = FakeInferenceClient()

    _, metadata = extract_structured(
        [str(image)], markdown_text="\n\n---\n\n", inference_client=client
    )

    assert client.requests[0].images
    assert metadata["source"] == "page_images_first_page"


def test_extract_structured_ignores_html_only_markdown_after_cleanup(tmp_path) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")
    client = FakeInferenceClient()

    _, metadata = extract_structured(
        [str(image)], markdown_text="<div><br/></div><table></table>", inference_client=client
    )

    assert client.requests[0].images
    assert metadata["source"] == "page_images_first_page"


def test_extract_structured_passes_model_override_to_provider_neutral_request(tmp_path) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")
    client = FakeInferenceClient(model=None)

    _, metadata = extract_structured(
        [str(image)], model="manual-model", inference_client=client
    )

    assert client.requests[0].model == "manual-model"
    assert metadata["model"] == "manual-model"


def test_extract_structured_surfaces_provider_errors(tmp_path) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")
    client = FakeInferenceClient(error=ProviderRequestError("provider offline"))

    with pytest.raises(ProviderRequestError, match="provider offline"):
        extract_structured([str(image)], inference_client=client)


def test_extract_structured_rejects_malformed_provider_json(tmp_path) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")
    client = FakeInferenceClient(text="not json")

    with pytest.raises(RuntimeError, match="Could not parse structured JSON response"):
        extract_structured([str(image)], inference_client=client)
