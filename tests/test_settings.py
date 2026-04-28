from __future__ import annotations

from pathlib import Path

import pytest

from my_ocr.settings import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    resolve_inference_provider_config,
)


def test_resolve_inference_provider_config_uses_ollama_defaults_when_config_is_missing(
    tmp_path,
) -> None:
    config = resolve_inference_provider_config(tmp_path / "missing.yaml")

    assert config.provider == "ollama"
    assert config.model == DEFAULT_OLLAMA_MODEL
    assert config.base_url == DEFAULT_OLLAMA_BASE_URL
    assert config.endpoint == DEFAULT_OLLAMA_ENDPOINT
    assert config.num_ctx == DEFAULT_OLLAMA_NUM_CTX
    assert config.max_tokens is None
    assert config.api_key is None
    assert config.extra == {}


def test_resolve_inference_provider_config_reads_ollama_yaml_overrides(tmp_path) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  inference:\n"
        "    provider: ollama\n"
        "    model: glm-ocr:test\n"
        "    base_url: http://ollama.example:1234\n"
        "    num_ctx: 4096\n"
        "    max_tokens: 256\n"
        "    extra:\n"
        "      keep_alive: 5m\n",
        encoding="utf-8",
    )

    config = resolve_inference_provider_config(config_path)

    assert config.provider == "ollama"
    assert config.model == "glm-ocr:test"
    assert config.base_url == "http://ollama.example:1234"
    assert config.endpoint == "http://ollama.example:1234/api/generate"
    assert config.num_ctx == 4096
    assert config.max_tokens == 256
    assert config.extra == {"keep_alive": "5m"}


def test_resolve_inference_provider_config_maps_openai_compatible_vllm_fields(
    tmp_path,
) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  inference:\n"
        "    provider: openai_compatible\n"
        "    model: qwen2-vl\n"
        "    base_url: http://vllm.example:8000/v1\n"
        "    api_key: test-key\n"
        "    max_tokens: 1024\n"
        "    extra:\n"
        "      top_k: 5\n"
        "      repetition_penalty: 1.05\n",
        encoding="utf-8",
    )

    config = resolve_inference_provider_config(config_path)

    assert config.provider == "openai_compatible"
    assert config.model == "qwen2-vl"
    assert config.base_url == "http://vllm.example:8000/v1"
    assert config.endpoint == "http://vllm.example:8000/v1/chat/completions"
    assert config.api_key == "test-key"
    assert config.max_tokens == 1024
    assert config.num_ctx is None
    assert config.extra == {"top_k": 5, "repetition_penalty": 1.05}
    assert "test-key" not in repr(config)


@pytest.mark.parametrize(
    ("provider", "model", "base_url", "expected_endpoint"),
    [
        (
            "openai_compatible",
            "qwen2-vl",
            "http://vllm.example:8000/v1",
            "http://vllm.example:8000/v1/chat/completions",
        ),
        (
            "ollama",
            "custom-ocr:stable",
            "http://ollama.example:11434",
            "http://ollama.example:11434/api/generate",
        ),
    ],
)
def test_resolve_inference_provider_config_keeps_configured_model_over_caller_default(
    tmp_path,
    provider: str,
    model: str,
    base_url: str,
    expected_endpoint: str,
) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  inference:\n"
        f"    provider: {provider}\n"
        f"    model: {model}\n"
        f"    base_url: {base_url}\n",
        encoding="utf-8",
    )

    config = resolve_inference_provider_config(
        config_path,
        default_model="glm-ocr-8k:latest",
    )

    assert config.model == model
    assert config.endpoint == expected_endpoint


def test_resolve_inference_provider_config_raises_on_invalid_yaml(tmp_path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("pipeline:\n  inference: [broken\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid OCR config YAML"):
        resolve_inference_provider_config(config_path)


def test_resolve_inference_provider_config_rejects_unknown_provider(tmp_path) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        "pipeline:\n  inference:\n    provider: hosted_magic\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="provider must be one of"):
        resolve_inference_provider_config(config_path)


@pytest.mark.parametrize("value", ["'1024'", "1.5", "true", "0", "-1"])
def test_resolve_inference_provider_config_rejects_invalid_numeric_fields(
    tmp_path,
    value,
) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  inference:\n"
        "    provider: ollama\n"
        "    max_tokens: "
        f"{value}\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="max_tokens"):
        resolve_inference_provider_config(config_path)


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
