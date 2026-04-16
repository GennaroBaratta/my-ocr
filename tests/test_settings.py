from __future__ import annotations

import pytest

from free_doc_extract.settings import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    resolve_ocr_api_client,
)


def test_resolve_ocr_api_client_uses_defaults_when_config_is_missing(tmp_path) -> None:
    model, endpoint, num_ctx = resolve_ocr_api_client(tmp_path / "missing.yaml")

    assert model == DEFAULT_OLLAMA_MODEL
    assert endpoint == DEFAULT_OLLAMA_ENDPOINT
    assert num_ctx == DEFAULT_OLLAMA_NUM_CTX


def test_resolve_ocr_api_client_reads_yaml_overrides(tmp_path) -> None:
    config = tmp_path / "local.yaml"
    config.write_text(
        "pipeline:\n"
        "  ocr_api:\n"
        "    model: glm-ocr:test\n"
        "    api_host: ollama.example\n"
        "    api_port: 1234\n"
        "    api_path: /api/generate\n"
        "    num_ctx: 4096\n",
        encoding="utf-8",
    )

    model, endpoint, num_ctx = resolve_ocr_api_client(config)

    assert model == "glm-ocr:test"
    assert endpoint == "http://ollama.example:1234/api/generate"
    assert num_ctx == 4096


def test_resolve_ocr_api_client_raises_on_invalid_yaml(tmp_path) -> None:
    config = tmp_path / "broken.yaml"
    config.write_text("pipeline:\n  ocr_api: [broken\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid OCR config YAML"):
        resolve_ocr_api_client(config)
