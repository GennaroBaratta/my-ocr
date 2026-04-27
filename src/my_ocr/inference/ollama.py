from __future__ import annotations

import base64
from http.client import IncompleteRead, RemoteDisconnected
import json
from pathlib import Path
import socket
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from my_ocr.inference.contracts import (
    InferenceImage,
    InferenceRequest,
    InferenceResponse,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from my_ocr.settings import DEFAULT_OLLAMA_TIMEOUT_SECONDS

ALLOWED_URL_SCHEMES = {"http", "https"}
OLLAMA_RESERVED_EXTRA_FIELDS = {"format", "images", "model", "options", "prompt", "stream"}


class OllamaClient:
    """Non-streaming Ollama /api/generate adapter."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        opener: Callable[..., Any] = urlopen,
        default_max_tokens: int | None = None,
        default_num_ctx: int | None = None,
        default_extra: dict[str, Any] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._timeout = timeout
        self._opener = opener
        self._default_max_tokens = default_max_tokens
        self._default_num_ctx = default_num_ctx
        self._default_extra = dict(default_extra or {})

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        payload = build_generate_payload(
            request,
            default_model=self._model,
            default_max_tokens=self._default_max_tokens,
            default_num_ctx=self._default_num_ctx,
            default_extra=self._default_extra,
        )
        try:
            body = post_json(
                endpoint=self._endpoint,
                payload=payload,
                timeout=self._timeout,
                error_prefix="Ollama inference request",
                opener=self._opener,
            )
        except ProviderTimeoutError:
            raise
        except RuntimeError as exc:
            raise ProviderRequestError(str(exc)) from exc
        if "response" not in body:
            raise ProviderResponseError(f"Ollama response missing 'response' field: {body}")
        return InferenceResponse(text=str(body["response"]), raw=body, model=body.get("model"))


def build_generate_payload(
    request: InferenceRequest,
    *,
    default_model: str,
    default_max_tokens: int | None = None,
    default_num_ctx: int | None = None,
    default_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model or default_model,
        "prompt": request.prompt,
        "stream": False,
    }
    default_extra = dict(default_extra or {})
    request_extra = dict(request.extra)

    if request.images:
        payload["images"] = [_image_to_base64(image) for image in request.images]
    schema = request.json_schema or (
        request.structured_output.schema if request.structured_output else None
    )
    if schema is not None:
        payload["format"] = schema

    default_options = _pop_mapping(default_extra, "options")
    request_options = _pop_mapping(request_extra, "options")
    options = _merge_options(
        default_options,
        request_options,
        default_max_tokens=default_max_tokens,
        default_num_ctx=default_num_ctx,
        request=request,
    )
    if options:
        payload["options"] = options

    extra = dict(default_extra)
    extra.update(request_extra)
    _reject_reserved_extra_fields(extra)
    payload.update(extra)
    return payload


def _pop_mapping(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if not isinstance(value, dict):
        return None
    return payload.pop(key)


def _reject_reserved_extra_fields(extra: dict[str, Any]) -> None:
    collisions = sorted(key for key in extra if key in OLLAMA_RESERVED_EXTRA_FIELDS)
    if collisions:
        raise ValueError(
            "Ollama extra fields cannot override adapter-owned fields: "
            f"{', '.join(collisions)}"
        )


def _merge_options(
    default_options: dict[str, Any] | None,
    request_options: dict[str, Any] | None,
    *,
    default_max_tokens: int | None,
    default_num_ctx: int | None,
    request: InferenceRequest,
) -> dict[str, Any]:
    options = dict(default_options or {})
    if default_num_ctx is not None:
        options["num_ctx"] = default_num_ctx
    max_tokens = request.max_tokens if request.max_tokens is not None else default_max_tokens
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request_options is not None:
        options.update(request_options)
    return options


def _image_to_base64(image: InferenceImage) -> str:
    if image.path is not None:
        return encode_image_file(image.path)
    return base64.b64encode(image.data or b"").decode("utf-8")


def encode_image_file(image_path: str | Path) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def post_json(
    *,
    endpoint: str,
    payload: dict[str, Any],
    timeout: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    error_prefix: str = "Ollama request",
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    request = Request(
        _validate_endpoint(endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with opener(request, timeout=timeout) as response:  # nosec B310
            body = json.loads(response.read().decode("utf-8"))
    except socket.timeout as exc:
        raise ProviderTimeoutError(f"{error_prefix} timed out") from exc
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{error_prefix} failed: {exc.code} {detail}") from exc
    except (
        RemoteDisconnected,
        IncompleteRead,
        ConnectionAbortedError,
        ConnectionResetError,
    ) as exc:
        raise RuntimeError(f"{error_prefix} failed: remote disconnected") from exc
    except URLError as exc:
        if isinstance(
            exc.reason,
            (
                ConnectionAbortedError,
                ConnectionResetError,
                BrokenPipeError,
                IncompleteRead,
                RemoteDisconnected,
            ),
        ):
            raise RuntimeError(f"{error_prefix} failed: remote disconnected") from exc
        raise RuntimeError(f"Could not reach Ollama at {endpoint}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{error_prefix} returned invalid JSON") from exc

    if not isinstance(body, dict):
        raise RuntimeError(f"{error_prefix} returned a non-object response")
    return body


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme not in ALLOWED_URL_SCHEMES or not parsed.netloc:
        raise ValueError(
            f"Unsupported Ollama endpoint scheme: {endpoint!r}. Expected http:// or https://"
        )
    return endpoint


__all__ = [
    "ALLOWED_URL_SCHEMES",
    "OllamaClient",
    "build_generate_payload",
    "encode_image_file",
    "post_json",
]
