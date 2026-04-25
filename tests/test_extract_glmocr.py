from __future__ import annotations

import json
from urllib.error import URLError

import pytest

from my_ocr.settings import DEFAULT_OLLAMA_MODEL
from my_ocr.extraction.structured import (
    _build_structured_markdown_payload,
    _clean_structured_input_text,
    build_structured_prompt,
    _parse_response_json,
    extract_structured,
    save_structured_result,
)


def test_parse_response_json_handles_fenced_json() -> None:
    parsed = _parse_response_json('```json\n{"title": "Demo"}\n```')
    assert parsed == {"title": "Demo"}


def test_parse_response_json_rejects_non_object() -> None:
    with pytest.raises(RuntimeError, match="was not a JSON object"):
        _parse_response_json("[1, 2, 3]")


def test_extract_structured_parses_response_and_returns_metadata(tmp_path, monkeypatch) -> None:
    image1 = tmp_path / "page-0001.png"
    image2 = tmp_path / "page-0002.png"
    image1.write_bytes(b"fake image 1")
    image2.write_bytes(b"fake image 2")

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {
                    "model": DEFAULT_OLLAMA_MODEL,
                    "created_at": "2026-01-01T00:00:00Z",
                    "done": True,
                    "response": '```json\n{"document_type": "report", "title": "Sample", "authors": [], "institution": "", "date": "", "language": "en", "summary_line": "Hi"}\n```',
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("my_ocr.extraction.structured.urlopen", fake_urlopen)

    parsed, metadata = extract_structured(
        [str(image1), str(image2)], endpoint="http://ollama.test/api/generate"
    )

    assert len(captured["body"]["images"]) == 1
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["num_ctx"] == 8192
    assert parsed["title"] == "Sample"
    assert metadata["done"] is True
    assert metadata["source"] == "page_images_first_page"
    assert metadata["image_paths"] == [str(image1)]
    assert metadata["input_page_count"] == 2
    assert metadata["used_page_count"] == 1
    assert metadata["_raw_body"]["response"].startswith("```json")


def test_extract_structured_uses_markdown_text_when_provided(tmp_path, monkeypatch) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps({"response": '{"document_type":"report","authors":[]}'}).encode(
                "utf-8"
            )

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("my_ocr.extraction.structured.urlopen", fake_urlopen)

    parsed, metadata = extract_structured([str(image)], markdown_text="# OCR\nSome text")

    assert "images" not in captured["body"]
    assert "OCR text:" in captured["body"]["prompt"]
    assert captured["body"]["options"]["num_ctx"] == 8192
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


def test_structured_markdown_payload_uses_cleaned_text() -> None:
    payload = _build_structured_markdown_payload(
        "<div>  A   B </div><table><tr><td>X</td><td>Y</td></tr></table>",
        model="demo",
    )

    assert payload["prompt"].endswith("OCR text:\nA B X | Y")
    assert payload["format"] is not None


def test_extract_structured_ignores_separator_only_markdown(tmp_path, monkeypatch) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps({"response": '{"document_type":"report","authors":[]}'}).encode(
                "utf-8"
            )

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("my_ocr.extraction.structured.urlopen", fake_urlopen)

    _, metadata = extract_structured([str(image)], markdown_text="\n\n---\n\n")

    assert "images" in captured["body"]
    assert metadata["source"] == "page_images_first_page"


def test_extract_structured_ignores_html_only_markdown_after_cleanup(tmp_path, monkeypatch) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps({"response": '{"document_type":"report","authors":[]}'}).encode(
                "utf-8"
            )

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("my_ocr.extraction.structured.urlopen", fake_urlopen)

    _, metadata = extract_structured([str(image)], markdown_text="<div><br/></div><table></table>")

    assert "images" in captured["body"]
    assert metadata["source"] == "page_images_first_page"


def test_extract_structured_defaults_model_and_endpoint_from_config(tmp_path, monkeypatch) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")
    config = tmp_path / "local.yaml"
    config.write_text(
        "pipeline:\n  ocr_api:\n    model: glm-structured:test\n    api_host: ollama.example\n    api_port: 1234\n    api_path: /api/generate\n",
        encoding="utf-8",
    )

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps({"response": '{"document_type":"report","authors":[]}'}).encode(
                "utf-8"
            )

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["endpoint"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr("my_ocr.extraction.structured.urlopen", fake_urlopen)

    extract_structured([str(image)], config_path=config)

    assert captured["body"]["model"] == "glm-structured:test"
    assert captured["endpoint"] == "http://ollama.example:1234/api/generate"


def test_extract_structured_prefers_explicit_over_config(tmp_path, monkeypatch) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")
    config = tmp_path / "local.yaml"
    config.write_text(
        "pipeline:\n  ocr_api:\n    model: glm-structured:test\n    api_host: ollama.example\n    api_port: 1234\n    api_path: /api/generate\n",
        encoding="utf-8",
    )

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps({"response": '{"document_type":"report","authors":[]}'}).encode(
                "utf-8"
            )

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["endpoint"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr("my_ocr.extraction.structured.urlopen", fake_urlopen)

    extract_structured(
        [str(image)],
        config_path=config,
        model="manual-model",
        endpoint="http://manual.example/api/generate",
    )

    assert captured["body"]["model"] == "manual-model"
    assert captured["endpoint"] == "http://manual.example/api/generate"


def test_extract_structured_surfaces_request_errors(tmp_path, monkeypatch) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")

    def fake_urlopen(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr("my_ocr.extraction.structured.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        extract_structured([str(image)])


def test_extract_structured_rejects_missing_response_field(tmp_path, monkeypatch) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps({"model": DEFAULT_OLLAMA_MODEL}).encode("utf-8")

    monkeypatch.setattr(
        "my_ocr.extraction.structured.urlopen",
        lambda request, timeout: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="missing 'response'"):
        extract_structured([str(image)])


def test_save_structured_result_writes_prediction_files(tmp_path) -> None:
    save_structured_result(
        tmp_path,
        {"title": "Sample"},
        {"model": DEFAULT_OLLAMA_MODEL, "_raw_body": {"response": '{"title":"Sample"}'}},
    )

    pred_dir = tmp_path / "extraction"
    assert json.loads((pred_dir / "structured.json").read_text(encoding="utf-8")) == {
        "title": "Sample"
    }
    assert json.loads((pred_dir / "structured_meta.json").read_text(encoding="utf-8")) == {
        "model": DEFAULT_OLLAMA_MODEL
    }
    assert json.loads((pred_dir / "structured_raw.json").read_text(encoding="utf-8")) == {
        "response": '{"title":"Sample"}'
    }
