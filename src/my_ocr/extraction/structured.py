from __future__ import annotations

from pathlib import Path
from typing import Any

from my_ocr.domain import PageRef
from my_ocr.domain import StructuredExtractionOptions
from my_ocr.domain.document import DocumentFields, JSON_SCHEMA
from my_ocr.inference import (
    InferenceClient,
    InferenceImage,
    InferenceRequest,
    InferenceResponse,
    OllamaClient,
    OpenAICompatibleClient,
    StructuredOutputRequest,
)
from my_ocr.settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OLLAMA_MODEL,
    InferenceProviderConfig,
    resolve_inference_provider_config,
)

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
    inference_client: InferenceClient | None = None,
    inference_config: InferenceProviderConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if markdown_text is None and not image_paths:
        raise ValueError("image_paths cannot be empty")

    resolved_config = inference_config or resolve_inference_provider_config(
        config_path or DEFAULT_CONFIG_PATH,
        default_model=DEFAULT_MODEL,
    )
    client = inference_client
    if client is None or endpoint is not None:
        client = _build_inference_client(resolved_config, endpoint_override=endpoint)

    source = "page_images_first_page"
    input_length: int | None = None
    input_page_count = len(image_paths)
    used_page_count = 0
    image_paths_metadata: list[str] = []
    cleaned_markdown_text = _clean_structured_input_text(markdown_text or "")

    if _has_meaningful_markdown(cleaned_markdown_text):
        request = build_structured_markdown_request(
            markdown_text or "",
            model=model,
        )
        source = "ocr_markdown"
        input_length = len(cleaned_markdown_text)
        input_page_count = len(image_paths)
        used_page_count = 0
    else:
        request_image_paths = image_paths[:1]
        request = build_structured_image_request(
            request_image_paths,
            model=model,
        )
        source = "page_images_first_page"
        input_length = None
        input_page_count = len(image_paths)
        used_page_count = len(request_image_paths)
        image_paths_metadata = request_image_paths

    response = client.generate(request)
    parsed = DocumentFields.from_mapping(_parse_response_json(response.text)).to_dict()
    metadata = _build_metadata(
        response=response,
        fallback_model=model or resolved_config.model,
        source=source,
        input_length=input_length,
        image_paths=image_paths_metadata,
        input_page_count=input_page_count,
        used_page_count=used_page_count,
    )
    return parsed, metadata


def build_structured_markdown_request(
    markdown_text: str,
    *,
    model: str | None = None,
) -> InferenceRequest:
    cleaned_text = _clean_structured_input_text(markdown_text)
    return _structured_request(
        prompt=f"{build_structured_prompt()}\n\nOCR text:\n{cleaned_text}",
        model=model,
    )


def build_structured_image_request(
    image_paths: list[str],
    *,
    model: str | None = None,
) -> InferenceRequest:
    return _structured_request(
        prompt=build_structured_prompt(),
        model=model,
        images=tuple(InferenceImage(path=image_path) for image_path in image_paths),
    )


def _structured_request(
    *,
    prompt: str,
    model: str | None,
    images: tuple[InferenceImage, ...] = (),
) -> InferenceRequest:
    return InferenceRequest(
        prompt=prompt,
        model=model,
        images=images,
        structured_output=StructuredOutputRequest(
            schema=JSON_SCHEMA,
            name="document_fields",
            strict=True,
        ),
    )


def _build_metadata(
    *,
    response: InferenceResponse,
    fallback_model: str,
    source: str,
    input_length: int | None,
    image_paths: list[str],
    input_page_count: int,
    used_page_count: int,
) -> dict[str, Any]:
    return {
        "model": response.model or fallback_model,
        "source": source,
        "input_length": input_length,
        "image_paths": image_paths,
        "input_page_count": input_page_count,
        "used_page_count": used_page_count,
        RAW_BODY_METADATA_KEY: response.raw,
    }


def _build_inference_client(
    config: InferenceProviderConfig,
    *,
    endpoint_override: str | None = None,
) -> InferenceClient:
    endpoint = endpoint_override or config.endpoint
    if config.provider == "ollama":
        return OllamaClient(
            endpoint=endpoint,
            model=config.model,
            timeout=config.timeout_seconds,
            default_max_tokens=config.max_tokens,
            default_num_ctx=config.num_ctx,
            default_extra=config.extra,
        )
    if config.provider == "openai_compatible":
        return OpenAICompatibleClient(
            endpoint=endpoint,
            model=config.model,
            timeout=config.timeout_seconds,
            api_key=config.api_key,
            default_max_tokens=config.max_tokens,
            default_extra=config.extra,
        )
    raise RuntimeError(f"Unsupported inference provider: {config.provider}")


class ProviderNeutralStructuredExtractor:
    def __init__(
        self,
        *,
        inference_client: InferenceClient | None = None,
        inference_config: InferenceProviderConfig | None = None,
    ) -> None:
        self._inference_client = inference_client
        self._inference_config = inference_config

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
            inference_client=self._inference_client,
            inference_config=self._inference_config,
        )


OllamaStructuredExtractor = ProviderNeutralStructuredExtractor

__all__ = [
    "ProviderNeutralStructuredExtractor",
    "OllamaStructuredExtractor",
    "RAW_BODY_METADATA_KEY",
    "build_structured_image_request",
    "build_structured_markdown_request",
    "build_structured_prompt",
    "extract_structured",
]
