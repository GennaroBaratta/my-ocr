from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RUN_ROOT = "data/runs"
DEFAULT_CONFIG_PATH = "config/local.yaml"
DEFAULT_LAYOUT_DEVICE = "cuda"
DEFAULT_INFERENCE_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "glm-ocr-8k:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_API_PATH = "/api/generate"
DEFAULT_OLLAMA_ENDPOINT = f"{DEFAULT_OLLAMA_BASE_URL}{DEFAULT_OLLAMA_API_PATH}"
DEFAULT_OLLAMA_NUM_CTX = 8192
DEFAULT_OLLAMA_KEEP_ALIVE = "15m"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 300
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OPENAI_COMPATIBLE_CHAT_PATH = "/chat/completions"
SUPPORTED_INFERENCE_PROVIDERS = frozenset({"ollama", "openai_compatible"})


@dataclass(frozen=True, slots=True)
class InferenceProviderConfig:
    provider: str
    model: str
    base_url: str
    endpoint: str
    api_key: str | None = field(default=None, repr=False)
    max_tokens: int | None = None
    num_ctx: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS


def resolve_inference_provider_config(
    config_path: str | Path,
    *,
    default_model: str = DEFAULT_OLLAMA_MODEL,
) -> InferenceProviderConfig:
    path = Path(config_path)
    if not path.exists():
        return _build_inference_config({}, default_model=default_model)

    return _build_inference_config(
        _load_inference_config(path),
        default_model=default_model,
    )


def resolve_ocr_api_client(
    config_path: str | Path,
    *,
    default_model: str = DEFAULT_OLLAMA_MODEL,
    default_endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
) -> tuple[str, str, int]:
    path = Path(config_path)
    if not path.exists():
        return default_model, default_endpoint, DEFAULT_OLLAMA_NUM_CTX

    config = resolve_inference_provider_config(path, default_model=default_model)
    if config.provider != "ollama":
        raise RuntimeError(
            "Current OCR fallback callers require pipeline.inference.provider: ollama; "
            "provider-neutral OCR execution is available through bootstrap services."
        )
    return config.model, config.endpoint, config.num_ctx or DEFAULT_OLLAMA_NUM_CTX


def _load_inference_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid OCR config YAML: {path}") from exc

    pipeline = payload.get("pipeline") if isinstance(payload, dict) else {}
    inference = pipeline.get("inference") if isinstance(pipeline, dict) else {}
    if inference is None:
        return {}
    if not isinstance(inference, dict):
        raise RuntimeError("Invalid inference config: pipeline.inference must be a mapping")
    return inference


def _build_inference_config(
    inference: dict[str, Any],
    *,
    default_model: str,
) -> InferenceProviderConfig:
    provider = str(inference.get("provider") or DEFAULT_INFERENCE_PROVIDER).strip().lower()
    if provider not in SUPPORTED_INFERENCE_PROVIDERS:
        raise RuntimeError(
            "Invalid inference config: provider must be one of "
            f"{sorted(SUPPORTED_INFERENCE_PROVIDERS)}"
        )

    extra = _optional_mapping(inference.get("extra"), field_name="pipeline.inference.extra")
    model = str(inference.get("model") or default_model)
    api_key = _optional_string(inference.get("api_key"))
    max_tokens = _optional_int(inference.get("max_tokens"), field_name="max_tokens")
    timeout_seconds = _optional_int(
        inference.get("timeout_seconds"), field_name="timeout_seconds"
    ) or DEFAULT_OLLAMA_TIMEOUT_SECONDS

    if provider == "ollama":
        base_url = str(inference.get("base_url") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        num_ctx = _optional_int(inference.get("num_ctx"), field_name="num_ctx")
        endpoint = _join_endpoint(base_url, DEFAULT_OLLAMA_API_PATH)
        return InferenceProviderConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            endpoint=endpoint,
            api_key=api_key,
            max_tokens=max_tokens,
            num_ctx=num_ctx or DEFAULT_OLLAMA_NUM_CTX,
            extra=extra,
            timeout_seconds=timeout_seconds,
        )

    base_url = str(inference.get("base_url") or DEFAULT_OPENAI_COMPATIBLE_BASE_URL).rstrip("/")
    endpoint = _join_endpoint(base_url, DEFAULT_OPENAI_COMPATIBLE_CHAT_PATH)
    return InferenceProviderConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        endpoint=endpoint,
        api_key=api_key,
        max_tokens=max_tokens,
        num_ctx=_optional_int(inference.get("num_ctx"), field_name="num_ctx"),
        extra=extra,
        timeout_seconds=timeout_seconds,
    )


def _join_endpoint(base_url: str, suffix: str) -> str:
    clean_base = base_url.rstrip("/")
    clean_suffix = suffix if suffix.startswith("/") else f"/{suffix}"
    if clean_base.endswith(clean_suffix):
        return clean_base
    return f"{clean_base}{clean_suffix}"


def _optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: Any, *, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"Invalid inference config: {field_name} must be an integer")
    if value <= 0:
        raise RuntimeError(f"Invalid inference config: {field_name} must be positive")
    return value


def _optional_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid inference config: {field_name} must be a mapping")
    return dict(value)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_INFERENCE_PROVIDER",
    "DEFAULT_LAYOUT_DEVICE",
    "DEFAULT_OLLAMA_API_PATH",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_ENDPOINT",
    "DEFAULT_OLLAMA_KEEP_ALIVE",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_NUM_CTX",
    "DEFAULT_OLLAMA_TIMEOUT_SECONDS",
    "DEFAULT_OPENAI_COMPATIBLE_BASE_URL",
    "DEFAULT_OPENAI_COMPATIBLE_CHAT_PATH",
    "DEFAULT_RUN_ROOT",
    "InferenceProviderConfig",
    "SUPPORTED_INFERENCE_PROVIDERS",
    "resolve_inference_provider_config",
    "resolve_ocr_api_client",
]
