from __future__ import annotations

import base64
from http.client import IncompleteRead, RemoteDisconnected
import json
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .settings import DEFAULT_OLLAMA_TIMEOUT_SECONDS

ALLOWED_URL_SCHEMES = {"http", "https"}


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
