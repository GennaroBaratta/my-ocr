from __future__ import annotations

from http.client import IncompleteRead, RemoteDisconnected
from urllib.error import URLError

import pytest

from my_ocr.ocr.ollama_client import post_json


def test_post_json_wraps_remote_disconnect() -> None:
    def failing_opener(request, timeout):
        raise RemoteDisconnected("closed")

    with pytest.raises(RuntimeError, match="remote disconnected"):
        post_json(endpoint="http://localhost:11434/api/generate", payload={}, opener=failing_opener)


def test_post_json_wraps_incomplete_read() -> None:
    def failing_opener(request, timeout):
        raise IncompleteRead(b"partial")

    with pytest.raises(RuntimeError, match="remote disconnected"):
        post_json(endpoint="http://localhost:11434/api/generate", payload={}, opener=failing_opener)


def test_post_json_wraps_url_error() -> None:
    def failing_opener(request, timeout):
        raise URLError("offline")

    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        post_json(endpoint="http://localhost:11434/api/generate", payload={}, opener=failing_opener)


def test_post_json_wraps_url_error_disconnect_reason() -> None:
    def failing_opener(request, timeout):
        raise URLError(ConnectionResetError("reset"))

    with pytest.raises(RuntimeError, match="remote disconnected"):
        post_json(endpoint="http://localhost:11434/api/generate", payload={}, opener=failing_opener)
