from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class InferenceImage:
    """Image input for provider-neutral inference requests."""

    path: str | Path | None = None
    data: bytes | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if self.path is None and self.data is None:
            raise ValueError("InferenceImage requires either path or data")
        if self.path is not None and self.data is not None:
            raise ValueError("InferenceImage accepts path or data, not both")


@dataclass(frozen=True)
class StructuredOutputRequest:
    """Provider-neutral JSON/JSON-schema structured output request."""

    schema: dict[str, Any] | None = None
    name: str = "document_fields"
    strict: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceRequest:
    """Minimal request shape needed by OCR fallback and structured extraction."""

    prompt: str
    model: str | None = None
    images: tuple[InferenceImage, ...] = ()
    json_schema: dict[str, Any] | None = None
    structured_output: StructuredOutputRequest | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResponse:
    text: str
    raw: dict[str, Any]
    model: str | None = None


class InferenceError(RuntimeError):
    """Base project-owned inference failure."""


class ProviderRequestError(InferenceError):
    """Provider rejected the request or could not be reached."""


class ProviderTimeoutError(InferenceError):
    """Provider request timed out."""


class ProviderResponseError(InferenceError):
    """Provider returned malformed or unsupported response data."""


@runtime_checkable
class InferenceClient(Protocol):
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        ...


__all__ = [
    "InferenceClient",
    "InferenceError",
    "InferenceImage",
    "InferenceRequest",
    "InferenceResponse",
    "ProviderRequestError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "StructuredOutputRequest",
]
