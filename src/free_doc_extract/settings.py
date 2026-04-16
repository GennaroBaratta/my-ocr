from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RUN_ROOT = "data/runs"
DEFAULT_CONFIG_PATH = "config/local.yaml"
DEFAULT_LAYOUT_DEVICE = "cpu"
DEFAULT_OLLAMA_MODEL = "glm-ocr:latest"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_NUM_CTX = 8192
DEFAULT_OLLAMA_KEEP_ALIVE = "15m"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 300


@dataclass(frozen=True, slots=True)
class AppSettings:
    run_root: str = DEFAULT_RUN_ROOT
    ocr_config_path: str = DEFAULT_CONFIG_PATH
    layout_device: str = DEFAULT_LAYOUT_DEVICE
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    ollama_keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE
    ollama_timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    ollama_num_ctx: int = DEFAULT_OLLAMA_NUM_CTX


def resolve_ocr_api_client(
    config_path: str | Path,
    *,
    default_model: str = DEFAULT_OLLAMA_MODEL,
    default_endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
) -> tuple[str, str, int]:
    path = Path(config_path)
    if not path.exists():
        return default_model, default_endpoint, DEFAULT_OLLAMA_NUM_CTX

    loaded = _load_ocr_api_config(path)
    model = str(loaded.get("model") or default_model)
    endpoint = _build_endpoint(
        host=loaded.get("api_host"),
        port=loaded.get("api_port"),
        api_path=loaded.get("api_path"),
        default_endpoint=default_endpoint,
    )
    num_ctx = int(loaded.get("num_ctx") or DEFAULT_OLLAMA_NUM_CTX)
    return model, endpoint, num_ctx


def _load_ocr_api_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid OCR config YAML: {path}") from exc

    pipeline = payload.get("pipeline") if isinstance(payload, dict) else {}
    ocr_api = pipeline.get("ocr_api") if isinstance(pipeline, dict) else {}
    return ocr_api if isinstance(ocr_api, dict) else {}


def _build_endpoint(*, host: Any, port: Any, api_path: Any, default_endpoint: str) -> str:
    if host in (None, "") or port in (None, "") or api_path in (None, ""):
        return default_endpoint
    path = str(api_path)
    if not path.startswith("/"):
        path = f"/{path}"
    return f"http://{host}:{port}{path}"
