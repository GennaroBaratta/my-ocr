from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class JsonWriter(Protocol):
    def __call__(self, path: str | Path, payload: Any) -> None: ...


class DocumentNormalizer(Protocol):
    def __call__(self, input_path: str, run_dir: str | Path) -> list[str]: ...


class OcrEngine(Protocol):
    def __call__(self, page_paths: list[str], run_dir: str | Path, **kwargs: Any) -> dict[str, Any]: ...


class MarkdownExtractor(Protocol):
    def __call__(self, markdown: str) -> dict[str, Any]: ...


class StructuredExtractor(Protocol):
    def __call__(self, page_paths: list[str], **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]: ...
