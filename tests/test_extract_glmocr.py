from __future__ import annotations

import json
from urllib.error import URLError

import pytest

from free_doc_extract.extract_glmocr import (
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
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {
                    "model": "glm-ocr:latest",
                    "created_at": "2026-01-01T00:00:00Z",
                    "done": True,
                    "response": '```json\n{"document_type": "report", "title": "Sample", "authors": [], "institution": "", "date": "", "language": "en", "summary_line": "Hi"}\n```',
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("free_doc_extract.extract_glmocr.urlopen", fake_urlopen)

    parsed, metadata = extract_structured([str(image)], endpoint="http://ollama.test/api/generate")

    assert captured["body"]["images"] and captured["body"]["stream"] is False
    assert parsed["title"] == "Sample"
    assert metadata["done"] is True
    assert metadata["_raw_body"]["response"].startswith("```json")


def test_extract_structured_surfaces_request_errors(tmp_path, monkeypatch) -> None:
    image = tmp_path / "page-0001.png"
    image.write_bytes(b"fake image")

    def fake_urlopen(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr("free_doc_extract.extract_glmocr.urlopen", fake_urlopen)

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
            return json.dumps({"model": "glm-ocr:latest"}).encode("utf-8")

    monkeypatch.setattr(
        "free_doc_extract.extract_glmocr.urlopen", lambda request, timeout: FakeResponse()
    )

    with pytest.raises(RuntimeError, match="missing 'response'"):
        extract_structured([str(image)])


def test_save_structured_result_writes_prediction_files(tmp_path) -> None:
    save_structured_result(
        tmp_path,
        {"title": "Sample"},
        {"model": "glm-ocr:latest", "_raw_body": {"response": '{"title":"Sample"}'}},
    )

    pred_dir = tmp_path / "predictions"
    assert json.loads((pred_dir / "glmocr_structured.json").read_text(encoding="utf-8")) == {
        "title": "Sample"
    }
    assert json.loads((pred_dir / "glmocr_structured_meta.json").read_text(encoding="utf-8")) == {
        "model": "glm-ocr:latest"
    }
    assert json.loads((pred_dir / "glmocr_structured_raw.json").read_text(encoding="utf-8")) == {
        "response": '{"title":"Sample"}'
    }
