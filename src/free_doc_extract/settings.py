from __future__ import annotations

from dataclasses import dataclass


DEFAULT_RUN_ROOT = "data/runs"
DEFAULT_CONFIG_PATH = "config/local.yaml"
DEFAULT_LAYOUT_DEVICE = "cpu"
DEFAULT_OLLAMA_MODEL = "glm-ocr:latest"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
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
