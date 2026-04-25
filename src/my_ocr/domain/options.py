from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OcrRuntimeOptions:
    config_path: str = "config/local.yaml"
    layout_device: str = "cuda"
    layout_profile: str | None = "auto"
    model: str | None = None
    endpoint: str | None = None
    num_ctx: int | None = None


@dataclass(frozen=True, slots=True)
class StructuredExtractionOptions:
    config_path: str = "config/local.yaml"
    model: str | None = None
    endpoint: str | None = None
