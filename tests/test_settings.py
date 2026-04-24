from __future__ import annotations

from pathlib import Path

import pytest

from my_ocr.adapters.outbound.config.settings import (
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


def test_repo_local_glmocr_config_includes_formula_pipeline_mappings() -> None:
    glmocr_config = pytest.importorskip("glmocr.config")
    config_path = Path(__file__).resolve().parents[1] / "config" / "local.yaml"

    loaded = glmocr_config.load_config(config_path)

    assert loaded.pipeline.page_loader.task_prompt_mapping == {
        "text": "Text Recognition:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
    }
    assert loaded.pipeline.result_formatter.label_visualization_mapping == {
        "table": ["table"],
        "formula": ["formula", "display_formula", "inline_formula"],
        "image": ["chart", "image"],
        "text": [
            "abstract",
            "algorithm",
            "content",
            "doc_title",
            "figure_title",
            "paragraph_title",
            "reference_content",
            "text",
            "vertical_text",
            "vision_footnote",
            "seal",
            "formula_number",
        ],
    }
