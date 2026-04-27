from __future__ import annotations

from my_ocr.inference.contracts import (
    InferenceClient,
    InferenceError,
    InferenceImage,
    InferenceRequest,
    InferenceResponse,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    StructuredOutputRequest,
)
from my_ocr.inference.ollama import OllamaClient, encode_image_file, post_json
from my_ocr.inference.openai_compatible import OpenAICompatibleClient

__all__ = [
    "InferenceClient",
    "InferenceError",
    "InferenceImage",
    "InferenceRequest",
    "InferenceResponse",
    "OllamaClient",
    "OpenAICompatibleClient",
    "ProviderRequestError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "StructuredOutputRequest",
    "encode_image_file",
    "post_json",
]
