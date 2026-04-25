from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LayoutOptions:
    config_path: str = "config/local.yaml"
    layout_device: str = "cuda"
    layout_profile: str | None = "auto"


@dataclass(frozen=True, slots=True)
class OcrOptions:
    config_path: str = "config/local.yaml"
    layout_device: str = "cuda"
    layout_profile: str | None = "auto"


@dataclass(frozen=True, slots=True)
class StructuredExtractionOptions:
    config_path: str = "config/local.yaml"
    model: str | None = None
    endpoint: str | None = None
