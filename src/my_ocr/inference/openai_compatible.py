from __future__ import annotations

import base64
from http.client import IncompleteRead, RemoteDisconnected
import json
import mimetypes
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
from my_ocr.inference.ollama import ALLOWED_URL_SCHEMES
from my_ocr.settings import DEFAULT_OLLAMA_TIMEOUT_SECONDS

JSON_CONTENT_TYPE = "application/json"


class OpenAICompatibleClient:
    """Non-streaming OpenAI Chat Completions adapter for vLLM-compatible servers."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        opener: Callable[..., Any] = urlopen,
        api_key: str | None = None,
        default_max_tokens: int | None = None,
        default_extra: dict[str, Any] | None = None,
    ) -> None:
        self._endpoint = _validate_endpoint(endpoint)
        self._model = model
        self._timeout = timeout
        self._opener = opener
        self._api_key = api_key
        self._default_max_tokens = default_max_tokens
        self._default_extra = dict(default_extra or {})

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        payload = build_chat_completions_payload(
            request,
            default_model=self._model,
            default_max_tokens=self._default_max_tokens,
            default_extra=self._default_extra,
        )
        headers = {"Content-Type": JSON_CONTENT_TYPE}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        http_request = Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        body = _post_json(http_request, timeout=self._timeout, opener=self._opener)
        return parse_chat_completions_response(body)


def build_chat_completions_payload(
    request: InferenceRequest,
    *,
    default_model: str,
    default_max_tokens: int | None = None,
    default_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model or default_model,
        "messages": [
            {
                "role": "user",
                "content": _build_user_content(request.prompt, request.images),
            }
        ],
        "stream": False,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    max_tokens = request.max_tokens if request.max_tokens is not None else default_max_tokens
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    structured = request.structured_output
    schema = request.json_schema or (structured.schema if structured else None)
    if schema is not None:
        name = structured.name if structured else "document_fields"
        strict = structured.strict if structured else True
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": strict},
        }

    extra = dict(default_extra or {})
    extra.update(request.extra)
    if structured is not None:
        extra.update(structured.extra)
    if extra:
        _reject_guided_fields(extra)
        collisions = sorted(key for key in extra if key in payload)
        if collisions:
            raise ValueError(
                "OpenAI-compatible extras cannot override base payload keys: "
                f"{collisions}"
            )
        payload.update(extra)
    _reject_guided_fields(payload)
    return payload


def parse_chat_completions_response(body: dict[str, Any]) -> InferenceResponse:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderResponseError("OpenAI-compatible response missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ProviderResponseError("OpenAI-compatible response choice was not an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ProviderResponseError("OpenAI-compatible response missing message")
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        text = "".join(str(part) for part in text_parts)
    else:
        raise ProviderResponseError("OpenAI-compatible response missing text content")
    return InferenceResponse(text=text, raw=body, model=body.get("model"))


def _build_user_content(prompt: str, images: tuple[InferenceImage, ...]) -> str | list[dict[str, Any]]:
    if not images:
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}}
        for image in images
    )
    return content


def _image_to_data_url(image: InferenceImage) -> str:
    mime_type = image.mime_type
    if image.path is not None:
        path = Path(image.path)
        data = path.read_bytes()
        mime_type = mime_type or mimetypes.guess_type(path.name)[0]
    else:
        data = image.data or b""
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"


def _post_json(
    request: Request,
    *,
    timeout: int,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    try:
        with opener(request, timeout=timeout) as response:  # nosec B310
            body = json.loads(response.read().decode("utf-8"))
    except socket.timeout as exc:
        raise ProviderTimeoutError("OpenAI-compatible request timed out") from exc
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderRequestError(
            f"OpenAI-compatible request failed: {exc.code} {detail}"
        ) from exc
    except (RemoteDisconnected, IncompleteRead, ConnectionAbortedError, ConnectionResetError) as exc:
        raise ProviderRequestError("OpenAI-compatible request failed: remote disconnected") from exc
    except URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise ProviderTimeoutError("OpenAI-compatible request timed out") from exc
        raise ProviderRequestError("Could not reach OpenAI-compatible provider") from exc
    except json.JSONDecodeError as exc:
        raise ProviderResponseError("OpenAI-compatible response returned invalid JSON") from exc

    if not isinstance(body, dict):
        raise ProviderResponseError("OpenAI-compatible response returned a non-object body")
    return body


def _reject_guided_fields(payload: dict[str, Any]) -> None:
    guided_fields = sorted(key for key in payload if key.startswith("guided_"))
    if guided_fields:
        raise ValueError(f"Deprecated guided_* request fields are not supported: {guided_fields}")


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme not in ALLOWED_URL_SCHEMES or not parsed.netloc:
        raise ValueError(
            f"Unsupported OpenAI-compatible endpoint scheme: {endpoint!r}. "
            "Expected http:// or https://"
        )
    return endpoint


__all__ = [
    "OpenAICompatibleClient",
    "build_chat_completions_payload",
    "parse_chat_completions_response",
]
