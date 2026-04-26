from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from my_ocr.domain.document import JSON_SCHEMA
from my_ocr.inference.ollama import encode_image_file, post_json
from my_ocr.settings import DEFAULT_OLLAMA_KEEP_ALIVE, DEFAULT_OLLAMA_NUM_CTX

from .structured_prompt import build_structured_prompt, clean_structured_input_text


def build_structured_image_payload(
    image_paths: list[str], *, model: str, num_ctx: int
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": build_structured_prompt(),
        "images": [encode_image_file(Path(image_path)) for image_path in image_paths],
        "format": JSON_SCHEMA,
        "stream": False,
        "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": num_ctx},
    }


def build_structured_markdown_payload(
    markdown_text: str, *, model: str, num_ctx: int = DEFAULT_OLLAMA_NUM_CTX
) -> dict[str, Any]:
    cleaned_text = clean_structured_input_text(markdown_text)
    return {
        "model": model,
        "prompt": f"{build_structured_prompt()}\n\nOCR text:\n{cleaned_text}",
        "format": JSON_SCHEMA,
        "stream": False,
        "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": num_ctx},
    }


def request_structured_response(
    *,
    endpoint: str,
    payload: dict[str, Any],
    opener: Callable[..., Any],
) -> dict[str, Any]:
    return post_json(
        endpoint=endpoint,
        payload=payload,
        error_prefix="Ollama request",
        opener=opener,
    )


__all__ = [
    "build_structured_image_payload",
    "build_structured_markdown_payload",
    "request_structured_response",
]
