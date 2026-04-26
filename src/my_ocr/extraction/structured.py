from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.request import urlopen

from my_ocr.domain import PageRef
from my_ocr.domain import StructuredExtractionOptions
from my_ocr.domain.document import DocumentFields
from my_ocr.settings import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    resolve_ocr_api_client,
)

from .ollama_structured import (
    build_structured_image_payload as _build_structured_image_payload,
)
from .ollama_structured import (
    build_structured_markdown_payload as _build_structured_markdown_payload,
)
from .ollama_structured import request_structured_response
from .parse_json import parse_response_json as _parse_response_json
from .structured_prompt import build_structured_prompt
from .structured_prompt import clean_structured_input_text as _clean_structured_input_text
from .structured_prompt import has_meaningful_markdown as _has_meaningful_markdown

DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL
RAW_BODY_METADATA_KEY = "_raw_body"


def extract_structured(
    image_paths: list[str],
    *,
    markdown_text: str | None = None,
    config_path: str | Path | None = None,
    model: str | None = None,
    endpoint: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if markdown_text is None and not image_paths:
        raise ValueError("image_paths cannot be empty")

    resolved_model = model
    resolved_endpoint = endpoint
    resolved_num_ctx = DEFAULT_OLLAMA_NUM_CTX
    if config_path is not None:
        config_resolution = resolve_ocr_api_client(config_path)
        config_model, config_endpoint, config_num_ctx = config_resolution
        if resolved_model is None:
            resolved_model = config_model
        if resolved_endpoint is None:
            resolved_endpoint = config_endpoint
        resolved_num_ctx = config_num_ctx

    if resolved_model is None:
        resolved_model = DEFAULT_MODEL
    if resolved_endpoint is None:
        resolved_endpoint = DEFAULT_OLLAMA_ENDPOINT

    body: dict[str, Any] = {}
    source = "page_images_first_page"
    input_length: int | None = None
    input_page_count = len(image_paths)
    used_page_count = 0
    image_paths_metadata: list[str] = []
    cleaned_markdown_text = _clean_structured_input_text(markdown_text or "")

    if _has_meaningful_markdown(cleaned_markdown_text):
        body = request_structured_response(
            endpoint=resolved_endpoint,
            payload=_build_structured_markdown_payload(
                markdown_text or "", model=resolved_model, num_ctx=resolved_num_ctx
            ),
            opener=urlopen,
        )
        source = "ocr_markdown"
        input_length = len(cleaned_markdown_text)
        input_page_count = len(image_paths)
        used_page_count = 0
    else:
        request_image_paths = image_paths[:1]
        body = request_structured_response(
            endpoint=resolved_endpoint,
            payload=_build_structured_image_payload(
                request_image_paths, model=resolved_model, num_ctx=resolved_num_ctx
            ),
            opener=urlopen,
        )
        source = "page_images_first_page"
        input_length = None
        input_page_count = len(image_paths)
        used_page_count = len(request_image_paths)
        image_paths_metadata = request_image_paths

    if "response" not in body:
        raise RuntimeError(f"Ollama response missing 'response' field: {body}")

    parsed = DocumentFields.from_mapping(_parse_response_json(body["response"])).to_dict()
    metadata = {
        "model": body.get("model", resolved_model),
        "created_at": body.get("created_at"),
        "total_duration": body.get("total_duration"),
        "load_duration": body.get("load_duration"),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
        "done": body.get("done"),
        "source": source,
        "input_length": input_length,
        "image_paths": image_paths_metadata,
        "input_page_count": input_page_count,
        "used_page_count": used_page_count,
        RAW_BODY_METADATA_KEY: body,
    }
    return parsed, metadata


class OllamaStructuredExtractor:
    def extract(
        self,
        pages: list[PageRef],
        *,
        markdown_text: str | None,
        options: StructuredExtractionOptions,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return extract_structured(
            [str(page.path_for_io) for page in pages],
            markdown_text=markdown_text,
            config_path=options.config_path,
            model=options.model,
            endpoint=options.endpoint,
        )

__all__ = [
    "OllamaStructuredExtractor",
    "RAW_BODY_METADATA_KEY",
    "build_structured_prompt",
    "extract_structured",
]
